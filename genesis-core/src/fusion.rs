use crate::{IValue, SCALE, NanoModule, NeuronsSoA, SynapsesSoA};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpikingFusionModule {
    pub fusion_neuron_indices: Vec<usize>,
    pub vision_weight: IValue,
    pub text_weight: IValue,
    pub audio_weight: IValue,
}

impl SpikingFusionModule {
    pub fn new(fusion_neurons: Vec<usize>) -> Self {
        Self {
            fusion_neuron_indices: fusion_neurons,
            vision_weight: SCALE,
            text_weight: SCALE,
            audio_weight: SCALE,
        }
    }

    /// Gated Fusion: multimodal integration with mutual inhibition.
    /// Each modality input is scaled by its current weight.
    /// Cross-modal terms (bi-modal interactions) are added for non-linear amplification.
    /// If a single modality dominates (e.g. vision > 80% threshold), it suppresses the others
    /// to reduce noise and clarify focus.
    pub fn fuse_scalar(&self, v: IValue, t: IValue, a: IValue) -> IValue {
        let v_gate = (v as i64 * self.vision_weight as i64) >> 10;
        let t_gate = (t as i64 * self.text_weight as i64) >> 10;
        let a_gate = (a as i64 * self.audio_weight as i64) >> 10;

        let cross_term = ((v_gate * t_gate) + (t_gate * a_gate) + (a_gate * v_gate)) >> 10;
        let mut final_sum = v_gate + t_gate + a_gate + cross_term;

        if v_gate > 800 { final_sum -= (t_gate + a_gate) / 4; }
        if t_gate > 800 { final_sum -= (v_gate + a_gate) / 4; }

        let res = final_sum as i32;
        res.clamp(0, SCALE * 2)
    }

    pub fn fuse_gated(
        &self,
        vision: &[IValue],
        text: &[IValue],
        audio: &[IValue],
    ) -> Vec<IValue> {
        let max_len = vision.len().max(text.len()).max(audio.len());
        let mut fused = vec![0; max_len];
        for i in 0..max_len {
            let v = vision.get(i).cloned().unwrap_or(0);
            let t = text.get(i).cloned().unwrap_or(0);
            let a = audio.get(i).cloned().unwrap_or(0);
            fused[i] = self.fuse_scalar(v, t, a);
        }
        fused
    }
}

impl NanoModule for SpikingFusionModule {
    fn name(&self) -> &str { "fusion" }
    fn tier(&self) -> u32 { 1 }

    fn on_tick(&mut self, bus: &crate::InputBus, _previous_spikes: &[bool], _tick: u32) {
        let range = if self.fusion_neuron_indices.is_empty() {
             0..bus.proximal.len()
        } else {
             0..0
        };

        if range.end > 0 {
            for i in range {
                let v = bus.get_modality(crate::Modality::Vision, i);
                let t = bus.get_modality(crate::Modality::Text, i);
                let a = bus.get_modality(crate::Modality::Audio, i);

                let fused_val = self.fuse_scalar(v, t, a);
                if fused_val > 0 {
                    crate::InputBus::atomic_saturating_add(&bus.proximal[i], fused_val);
                    crate::InputBus::atomic_saturating_add(&bus.distal[i], fused_val / 2);
                }
            }
        } else {
            for &i in &self.fusion_neuron_indices {
                if i < bus.proximal.len() {
                    let v = bus.get_modality(crate::Modality::Vision, i);
                    let t = bus.get_modality(crate::Modality::Text, i);
                    let a = bus.get_modality(crate::Modality::Audio, i);
                    let fused_val = self.fuse_scalar(v, t, a);
                    crate::InputBus::atomic_saturating_add(&bus.proximal[i], fused_val);
                    crate::InputBus::atomic_saturating_add(&bus.distal[i], fused_val / 2);
                }
            }
        }
    }
    fn on_update_weights(&mut self, _neurons: &mut NeuronsSoA, _previous_spikes: &[bool], _current_spikes: &[bool], _tick: u32, _reward: Option<IValue>) {}
    fn on_night_phase(&mut self, _neurons: &mut NeuronsSoA, _synapses: &mut SynapsesSoA, _reward: Option<IValue>) {}
    fn box_clone(&self) -> Box<dyn NanoModule> { Box::new(self.clone()) }
    fn get_state(&self) -> Vec<u8> { bincode::serialize(self).unwrap_or_default() }
    fn set_state(&mut self, state: &[u8]) {
        if let Ok(new_self) = bincode::deserialize::<Self>(state) { *self = new_self; }
    }
    fn validate_state(&self, neurons: &NeuronsSoA) -> Result<(), crate::ModuleError> {
        let n_count = neurons.len();
        for &idx in &self.fusion_neuron_indices {
            if idx >= n_count {
                return Err(crate::ModuleError::ValidationFailed(format!("Fusion neuron index {} out of bounds (n_count: {})", idx, n_count)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InputBus;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_fusion_gating_logic() {
        let module = SpikingFusionModule::new(vec![0]);

        // Balanced input
        let res_balanced = module.fuse_scalar(500, 500, 0);
        // cross_term = (500*500) >> 10 = 244
        // sum = 500 + 500 + 244 = 1244
        assert!(res_balanced > 1000);

        // Vision dominance suppression
        let res_dominant = module.fuse_scalar(900, 400, 0);
        // v_gate = 900, t_gate = 400
        // cross = (900*400) >> 10 = 351
        // sum = 900 + 400 + 351 = 1651
        // suppression = 400 / 4 = 100
        // final = 1551
        assert!(res_dominant < 1651);
    }

    #[test]
    fn test_fusion_on_tick() {
        let mut module = SpikingFusionModule::new(vec![0]);
        let bus = InputBus::new(1);

        bus.set_modality(crate::Modality::Vision, 0, 1000);
        bus.set_modality(crate::Modality::Text, 0, 1000);

        module.on_tick(&bus, &[false], 1);

        assert!(bus.proximal[0].load(Ordering::Relaxed) > 0);
        assert!(bus.distal[0].load(Ordering::Relaxed) > 0);
    }
}
