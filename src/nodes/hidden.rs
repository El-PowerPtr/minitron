use super::{
    input::*, 
    output::*
};
use crate::conn::connect::Connect;
use crate::conn::connection::Connection;

pub struct HiddenNode <I: Out + Connect, O: In> {
    inputs: Vec<Connection<I,Self>>,
    val: f32,
    bias: f32,
    outputs: Vec<Connection<Self,O>>
}

impl <O: In, I: Out + Connect> Out for HiddenNode<I,O>{
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x + self.bias)
    }
}

impl <O: In, I: Out + Connect> In for HiddenNode<I,O> {
    fn recieve(&mut self, x: f32) {
        self.val += x;
    }
}


impl <O: In, I: Out + Connect> Connect for HiddenNode<I, O> {
    fn forward_prop(&mut self, x: f32) {
        if x > 0.0 {
            for n in &mut self.outputs {
                n.to.borrow_mut().recieve((x + self.bias) * n.weight);
            }
        }
        
    }
}

