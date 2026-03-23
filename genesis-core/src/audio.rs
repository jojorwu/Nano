use crate::{IValue, SCALE};
use rand::Rng;

pub struct SpikingAudioModule {
    pub sample_rate: u32,
    pub sensitivity: i32,
}

impl SpikingAudioModule {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sensitivity: 100,
        }
    }

    /// Rate Encoding: Convert PCM samples to spike potentials.
    pub fn rate_encode(&self, samples: &[i16]) -> Vec<IValue> {
        let mut potentials = Vec::with_capacity(samples.len());
        for &s in samples {
            let abs_s = s.abs() as i32;
            let val = (abs_s * SCALE) / 32767;
            potentials.push(val * self.sensitivity / 100);
        }
        potentials
    }

    /// Poisson Encoding: Generate stochastic spikes from audio amplitude.
    pub fn poisson_encode(&self, samples: &[i16]) -> Vec<bool> {
        let mut rng = rand::thread_rng();
        let mut spikes = Vec::with_capacity(samples.len());
        for &s in samples {
            let rate = (s.abs() as f32) / 32767.0;
            spikes.push(rng.gen::<f32>() < rate * (self.sensitivity as f32 / 100.0));
        }
        spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_encoding() {
        let audio = SpikingAudioModule::new(44100);
        let samples = vec![0, 16384, 32767];
        let potentials = audio.rate_encode(&samples);
        assert_eq!(potentials[0], 0);
        assert!(potentials[1] > 400 && potentials[1] < 600);
        assert!(potentials[2] >= 999);
    }

    #[test]
    fn test_audio_poisson() {
        let audio = SpikingAudioModule::new(44100);
        let samples = vec![32767; 100]; // Max amplitude
        let spikes = audio.poisson_encode(&samples);
        let count = spikes.iter().filter(|&&s| s).count();
        assert!(count > 0);
    }
}
