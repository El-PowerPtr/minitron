use super::{
    input::*,
    output::*,
    learn::*,
    label::Label,
};
use crate::{
    conn::connection::Connection,
    learning_rate::*,
    SharedRef,
    MultiRef,
};

pub struct OutputNode<I: Out,M: LearningRateManager,T> {
    learning_manager: SharedRef<M>,
    err: f32,
    bias: f32,
    val: f32,
    connections: Vec<Connection<I, Self>>,
    label: MultiRef<dyn Label<T>>,
}

impl <I: Out, M:LearningRateManager, T> Out for OutputNode<I,M,T> {
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x - self.bias)
    }
    #[inline]
    fn local_gradient(&self, err: f32) -> f32 {
        err * sigmoid_derivative(self.val + self.bias)
    }
    
    fn forward_prop(&mut self, x: f32) {
        self.label.borrow_mut().recieve(self.activation(x));
    }
}

impl <I: Out, M:LearningRateManager, T> In for OutputNode<I,M,T>  {
    fn recieve(&mut self, x: f32) {
        self.val += x
    }
}

impl <I: Out, M: LearningRateManager, T> Learn for OutputNode<I,M,T> {
    fn learn(&mut self){
        let local_gradient = self.local_gradient(self.err);
        self.bias -= self.learning_manager.borrow().learning_rate() * local_gradient;
    }
    
    fn get_feedback(&mut self, feedback: f32){
        self.err += feedback;
    }
}

impl <I: Out + Learn, M: LearningRateManager, T> BackProp for OutputNode <I,M,T>{
    fn send_feedback(&mut self){
        for i in &mut self.connections {
            i.get_feedback(self.err);
        };
    }
}