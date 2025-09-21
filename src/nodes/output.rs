use super::input::*;

pub (super) trait Out{
    fn activation(&self, x: f32) -> f32;
}

pub (super) trait Connect<N: In>{
    fn connect(&mut self, weight: f32, node: InRef<N>);
    fn forward_prop(&mut self,x: f32);
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + x.exp())
}