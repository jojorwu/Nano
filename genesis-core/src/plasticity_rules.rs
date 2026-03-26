use crate::{IValue, SCALE, Compartment, model::NeuronsSoA, config::NetworkConfig, NeuromodulationState};

/// Context passed to plasticity rules to improve flexibility and reduce argument count.
pub struct PlasticityContext<'a> {
    pub pre_spiked: bool,
    pub post_spiked: bool,
    pub backprop_signal: IValue,
    pub prediction_error: IValue,
    pub compartment: Compartment,
    pub reward: Option<IValue>,
    pub neuromodulation: NeuromodulationState,
    pub pre_last_spike: u32,
    pub post_last_spike: u32,
    pub current_tick: u32,
    pub post_index: usize,
    pub neurons: &'a NeuronsSoA,
    pub config: &'a NetworkConfig,
}

impl<'a> PlasticityContext<'a> {
    /// Calculates the combined modulation factor based on chemical state and SMBP.
    pub fn get_modulation_gain(&self) -> i64 {
        let neuromod = SCALE as i64 + self.neuromodulation.noradrenaline as i64;
        let dopamine = SCALE as i64 + self.neuromodulation.dopamine.abs() as i64;
        let smbp = if self.compartment != Compartment::Proximal {
            SCALE as i64 + self.backprop_signal as i64
        } else {
            SCALE as i64
        };
        (neuromod * dopamine * smbp) >> 20
    }

    pub fn is_rewarded(&self) -> bool {
        self.reward.is_some()
    }
}

/// Trait for weight update rules (e.g., GSOP, STDP)
pub trait PlasticityRule {
    fn apply(&self, weight: &mut IValue, ctx: &PlasticityContext);

    /// STC: Tagging phase. Instead of updating weight, we update the tag trace.
    fn tag(&self, tag: &mut IValue, timer: &mut u16, volatility: &mut u8, causality: &mut u8, ctx: &PlasticityContext) {
        if *volatility > 0 {
             use rand::Rng;
             let mut rng = rand::thread_rng();
             if rng.gen_range(0..255) <= *volatility {
                  let mut temp_weight = *tag;
                  self.apply(&mut temp_weight, ctx);

                  if temp_weight != *tag {
                       *tag = temp_weight;
                       *timer = 100; // Tag duration: 100 ticks
                       *volatility = volatility.saturating_sub(1);

                       if ctx.pre_spiked && ctx.post_spiked {
                            *causality = causality.saturating_add(5);
                       }
                  }
             }
        }
    }

    fn update_contrastive(&self, weight: &mut IValue, layer_correlation: IValue) {
        if layer_correlation > 512 {
             *weight = (*weight as i64 * (1024 - (layer_correlation / 10)) as i64 >> 10) as i32;
        }
    }
}

pub struct GsopRule {
    pub learning_rate: IValue,
}

impl PlasticityRule for GsopRule {
    fn apply(&self, weight: &mut IValue, ctx: &PlasticityContext) {
        let mut lr = match ctx.compartment {
            Compartment::Proximal => self.learning_rate,
            Compartment::Distal => self.learning_rate * 8 / 10,
            _ => self.learning_rate / 2,
        };

        if ctx.compartment == Compartment::Distal {
             let pc_factor = (ctx.prediction_error as i64 * SCALE as i64) >> 10;
             lr = (lr as i64 * (SCALE as i64 + pc_factor.abs()) >> 10) as i32;
        }

        let p_gate = ctx.neurons.plasticity_gate[ctx.post_index];
        let lr_final = (lr as i64 * ctx.get_modulation_gain() * p_gate as i64) >> 20;
        let lr_final = lr_final as i32;

        let reward_mod = if let Some(r) = ctx.reward { if r < 0 { -1 } else { 1 } } else { 1 };
        let lr_mod = (lr_final * reward_mod) as i32;

        let old_weight = *weight;
        if ctx.pre_spiked {
             if ctx.compartment == Compartment::Distal {
                  if ctx.prediction_error > 100 {
                       *weight = weight.saturating_add(lr_mod);
                  } else if ctx.prediction_error < -100 {
                       *weight = weight.saturating_sub(lr_mod);
                  }
             } else {
                  if ctx.post_spiked {
                       *weight = weight.saturating_add(lr_mod);
                  } else {
                       *weight = weight.saturating_sub(lr_mod / 2);
                  }
             }
        }
        crate::plasticity::clamp_and_preserve_sign_with_limit(weight, old_weight, ctx.config.plasticity.weight_clamp_limit);
    }
}
