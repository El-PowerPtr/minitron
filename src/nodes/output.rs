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

pub struct OutputNode {
    bias: f32,
    inputs: Vec<f32>
}

impl Out for OutputNode {
     fn activation(&self, x: f32) -> f32 {
        sigmoid(x) - self.bias
    }
}

impl In for OutputNode {
    fn recieve(&mut self, x: f32) {
        self.inputs.push(x)
    }
}