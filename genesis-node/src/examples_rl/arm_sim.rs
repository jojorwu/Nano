use genesis_core::rl::Environment;
use genesis_core::IValue;

pub struct RobotArmEnv {
    pub joint_angle: f32,
    pub target_angle: f32,
    pub velocity: f32,
}

impl RobotArmEnv {
    pub fn new() -> Self {
        Self { joint_angle: 0.0, target_angle: 0.785, velocity: 0.0 }
    }
}

impl Environment for RobotArmEnv {
    fn observation_space(&self) -> usize { 2 }
    fn action_space(&self) -> usize { 1 }

    fn reset(&mut self) -> Vec<IValue> {
        self.joint_angle = 0.0;
        self.velocity = 0.0;
        vec![(self.joint_angle * 1000.0) as IValue, (self.target_angle * 1000.0) as IValue]
    }

    fn step(&mut self, actions: &[bool]) -> (Vec<IValue>, IValue, bool) {
        let torque = if actions[0] { 0.05 } else { -0.05 };
        self.velocity += torque - self.velocity * 0.1;
        self.joint_angle += self.velocity;

        let error = (self.target_angle - self.joint_angle).abs();
        let reward = if error < 0.1 { 20 } else { -2 };
        let done = error > 2.0;

        (
            vec![(self.joint_angle * 1000.0) as IValue, (self.target_angle * 1000.0) as IValue],
            reward,
            done
        )
    }
}
