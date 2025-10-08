use super::{
    input::*,
    learn::*,
};
use crate::{
    conn::{
        connection::Connection,
        connect::Connect,
    },
    learning_rate::*,
    SharedRef,
};

pub trait Out{
    fn activation(&self, x: f32) -> f32;
    fn local_gradient(&self, err: f32) ->f32;
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid_derivative(x: f32) -> f32 {
    let neg_exp = (-x).exp();
    let base = 1.0 + neg_exp;
    - neg_exp / (base * base)
}

#[inline]
pub fn error(actual: f32, expected: f32) -> f32 {
    (expected - actual) * (expected - actual)
}
    
#[inline]
pub fn error_der(actual: f32, expected:f32) -> f32 {
    -2.0 * (expected - actual)
}

pub struct OutputNode<I: Out + Connect,M: LearningRateManager> {
    learning_manager: SharedRef<M>,
    err: f32,
    bias: f32,
    val: f32,
    connections: Vec<Connection<I, Self>>
}

impl <I: Out + Connect, M:LearningRateManager> Out for OutputNode<I,M> {
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x - self.bias)
    }
    #[inline]
    fn local_gradient(&self, err: f32) -> f32 {
        err * sigmoid_derivative(self.val + self.bias)
    }
}

impl <I: Out + Connect, M:LearningRateManager> In for OutputNode<I,M>  {
    fn recieve(&mut self, x: f32) {
        self.val += x
    }
}

impl <I: Out + Connect, M: LearningRateManager> Learn for OutputNode<I,M> {
    fn learn(&mut self){
        let local_gradient = self.local_gradient(self.err);
        self.bias -= self.learning_manager.borrow().learning_rate() * local_gradient;
    }
    
    fn get_feedback(&mut self, feedback: f32){
        self.err += feedback;
    }
}

impl <I: Out + Connect + Learn, M: LearningRateManager> BackProp for OutputNode <I,M>{
    fn send_feedback(&mut self){
        for i in &mut self.connections {
            i.get_feedback(self.err);
        };
    }
}