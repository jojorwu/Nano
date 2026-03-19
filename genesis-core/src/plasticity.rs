use crate::{Synapse, IValue};

pub struct StructuralPlasticityConfig {
    pub prune_threshold: IValue,
    pub grow_threshold: usize, // number of times fired together
    pub max_synapses: usize,
}

impl Default for StructuralPlasticityConfig {
    fn default() -> Self {
        Self {
            prune_threshold: 10, // weight < 0.01 -> prune
            grow_threshold: 5,
            max_synapses: 1000000,
        }
    }
}

pub fn prune_synapses(synapses: &mut Vec<Synapse>, threshold: IValue) -> usize {
    let initial_count = synapses.len();
    synapses.retain(|s| s.weight.abs() >= threshold);
    initial_count - synapses.len()
}

pub fn grow_synapse(
    synapses: &mut Vec<Synapse>,
    source: u32,
    target: u32,
    initial_weight: IValue,
    config: &StructuralPlasticityConfig
) -> bool {
    if synapses.len() >= config.max_synapses {
        return false;
    }

    // Don't duplicate if already exists (simplified)
    if synapses.iter().any(|s| s.source_index == source && s.target_index == target) {
        return false;
    }

    synapses.push(Synapse {
        source_index: source,
        target_index: target,
        weight: initial_weight,
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning() {
        let mut synapses = vec![
            Synapse { source_index: 0, target_index: 1, weight: 100 },
            Synapse { source_index: 1, target_index: 2, weight: 5 },
        ];
        let pruned = prune_synapses(&mut synapses, 10);
        assert_eq!(pruned, 1);
        assert_eq!(synapses.len(), 1);
    }
}
