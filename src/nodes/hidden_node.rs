use std::borrow::BorrowMut;
use super::{
    input::In, 
    output::*,
    learn::*,
    input_node::InputNode,
};
use crate::SharedRef;
use crate::learning_rate::LearningRateManager;
use crate::conn::connection::Connection;

pub struct HiddenNode <I: Out, M: LearningRateManager, O: In> {
    inputs: Vec<Connection<I,Self>>,
    learning_manager: SharedRef<M>,
    val: f32,
    bias: f32,
    acc_err: f32,
    outputs: Vec<Connection<Self,O>>
}

impl <I: Out, M: LearningRateManager, O: In> Out for HiddenNode<I,M,O>{
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x + self.bias)
    }
    #[inline]
    fn local_gradient(&self, err: f32) -> f32 {
        err * sigmoid_derivative(self.val + self.bias)
    }
    
    fn forward_prop(&mut self, x: f32) {
        let sent_x = self.activation(x);
            for n in &mut self.outputs {
                n.borrow_mut().send_fronward(sent_x);
            }
    }
}

impl <I: Out, M: LearningRateManager, O: In> In for HiddenNode<I,M,O> {
    fn recieve(&mut self, x: f32) {
        self.val += x;
    }
}

impl <I: Out, M: LearningRateManager, O: In> Learn for HiddenNode<I,M,O> {
    fn learn(&mut self){
        let local_gradient = self.local_gradient(self.acc_err);
        self.bias -= self.learning_manager.borrow().learning_rate() * local_gradient;
    }
    
    fn get_feedback(&mut self, feedback: f32){
        self.acc_err += feedback;
    }
}

impl <I: Out + Learn, M: LearningRateManager, O: In> BackProp for HiddenNode <I,M,O>{
    fn send_feedback(&mut self){
        for i in &mut self.inputs {
            i.get_feedback(self.acc_err);
        };
    }
}

impl <_Self: In, M: LearningRateManager, O: In> BackProp for HiddenNode <InputNode<_Self>,M,O>{
    fn send_feedback(&mut self){
        for i in &mut self.inputs {
            i.get_feedback(self.acc_err);
        };
    }
}