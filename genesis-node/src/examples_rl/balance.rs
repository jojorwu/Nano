use genesis_core::rl::Environment;
use genesis_core::IValue;

pub struct SimpleBalanceEnv {
    pub angle: f32,
    pub velocity: f32,
    pub steps: usize,
}

impl SimpleBalanceEnv {
    pub fn new() -> Self {
        Self { angle: 0.0, velocity: 0.0, steps: 0 }
    }
}

impl Environment for SimpleBalanceEnv {
    fn observation_space(&self) -> usize { 2 }
    fn action_space(&self) -> usize { 2 }

    fn reset(&mut self) -> Vec<IValue> {
        self.angle = 0.01;
        self.velocity = 0.0;
        self.steps = 0;
        vec![(self.angle * 1000.0) as IValue, (self.velocity * 1000.0) as IValue]
    }

    fn step(&mut self, actions: &[bool]) -> (Vec<IValue>, IValue, bool) {
        self.steps += 1;
        let force = if actions[0] { -0.01 } else if actions[1] { 0.01 } else { 0.0 };
        self.velocity += force + self.angle * 0.01;
        self.angle += self.velocity;

        let reward = if self.angle.abs() < 0.2 { 10 } else { -100 };
        let done = self.angle.abs() > 0.5 || self.steps > 500;

        (
            vec![(self.angle * 1000.0) as IValue, (self.velocity * 1000.0) as IValue],
            reward,
            done
        )
    }
}
