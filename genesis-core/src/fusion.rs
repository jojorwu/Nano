use crate::{IValue, SCALE};

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

    /// Gated Fusion: modalities weight each other.
    /// If one modality is strong, it can amplify or suppress others.
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

            // Calculate cross-modal gating factors
            // High vision activity might amplify text (e.g., reading)
            let v_gate = (v as i64 * self.vision_weight as i64) >> 10;
            let t_gate = (t as i64 * self.text_weight as i64) >> 10;
            let a_gate = (a as i64 * self.audio_weight as i64) >> 10;

            // Fused signal is a weighted sum with gating
            // We use a non-linear combination: (V*T + T*A + A*V) for high-order fusion
            let cross_term = ((v_gate * t_gate) + (t_gate * a_gate) + (a_gate * v_gate)) >> 10;

            fused[i] = (v_gate + t_gate + a_gate + cross_term) as i32;
            if fused[i] > SCALE * 2 { fused[i] = SCALE * 2; }
        }
        fused
    }
}
