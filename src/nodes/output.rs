use super::input::*;

pub (mod) trait Out{
    fn activation(x: f32) -> f32;
}

pub (mod) trait Connect<N: In>{
    fn connect(&mut self, weight: f32, node: InRef<N>);
    fn forward_prop(&self, f32);
}

#[inline]
fn sigmoid(x: &f32) -> f32 {
    1 / (1 + exp(x))
}