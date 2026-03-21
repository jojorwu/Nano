use crate::IValue;

pub struct SpikingFusionModule {
    pub fusion_neuron_indices: Vec<usize>,
}

impl SpikingFusionModule {
    pub fn new(fusion_neurons: Vec<usize>) -> Self {
        Self { fusion_neuron_indices: fusion_neurons }
    }

    /// Fuse inputs from different modalities by summing potential injections into shared neurons.
    pub fn fuse_injections(
        &self,
        vision_potentials: &[IValue],
        text_potentials: &[IValue],
        audio_potentials: &[IValue],
    ) -> Vec<IValue> {
        let max_len = vision_potentials.len()
            .max(text_potentials.len())
            .max(audio_potentials.len());

        let mut fused = vec![0; max_len];
        for i in 0..max_len {
            let v = vision_potentials.get(i).cloned().unwrap_or(0);
            let t = text_potentials.get(i).cloned().unwrap_or(0);
            let a = audio_potentials.get(i).cloned().unwrap_or(0);

            // Weighting modalities equally for now
            fused[i] = (v + t + a) / 3;
        }
        fused
    }
}
