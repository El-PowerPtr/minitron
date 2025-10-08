use super::{
    input::*,
    output::*,
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