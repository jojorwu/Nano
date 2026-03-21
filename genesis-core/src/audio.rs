use crate::{IValue, SCALE};

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

    /// Convert PCM samples (i16) to spike potentials.
    /// Uses simple rate encoding based on amplitude.
    pub fn rate_encode(&self, samples: &[i16]) -> Vec<IValue> {
        let mut potentials = Vec::with_capacity(samples.len());
        for &s in samples {
            // Map i16 amplitude to 0..SCALE
            let abs_s = s.abs() as i32;
            let val = (abs_s * SCALE) / 32767;
            potentials.push(val * self.sensitivity / 100);
        }
        potentials
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
}
