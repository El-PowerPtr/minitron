use super::{
    input::In, 
    output::{Out, sigmoid, sigmoid_derivative},
    learn::{Learn, BackProp},
    node::Node,
    input_node::InputNode,
};
use crate::{
    SharedRef,
    learning_rate::{
        LearningRateManager,
    },
    conn::connection::{
        Connection,
        ConnectsTo,
        ConnectsFrom,
    },
};

pub struct HiddenNode <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> {
    inputs: Vec<SharedRef<Connection<I,Self>>>,
    learning_manager: SharedRef<M>,
    val: f32,
    bias: f32,
    acc_err: f32,
    outputs: Vec<SharedRef<Connection<Self,O>>>
}

impl <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> Node<M> for HiddenNode<I,M,O>{
    fn new(bias: f32, manager: SharedRef<M>) -> Self{
        HiddenNode {
            inputs: vec![],
            learning_manager: manager.clone(),
            val: 0.0,
            bias,
            acc_err: 0.0,
            outputs: vec![]
        }
    }
}

impl <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> Out for HiddenNode<I,M,O>{
    #[inline]
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x + self.bias)
    }
    #[inline]
    fn local_gradient(&self, err: f32) -> f32 {
        err * sigmoid_derivative(self.val + self.bias)
    }
    
    fn forward_prop(&mut self) {
        let sent_x = self.activation(self.val);
            for n in &mut self.outputs {
                (**n).borrow_mut().send_fronward(sent_x);
            }
    }
}

impl <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> In for HiddenNode<I,M,O> {
    fn recieve(&mut self, x: f32) {
        self.val += x;
    }
}

impl <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> Learn for HiddenNode<I,M,O> {
    fn learn(&mut self){
        let local_gradient = self.local_gradient(self.acc_err);
        self.bias -= self.learning_manager.borrow().learning_rate() * local_gradient;
    }
    
    fn get_feedback(&mut self, feedback: f32){
        self.acc_err += feedback;
    }
}

impl <I: ConnectsTo + Learn, M: LearningRateManager, O: ConnectsFrom> BackProp for HiddenNode <I,M,O>{
    fn send_feedback(&mut self){
        for i in &mut self.inputs {
            (**i).borrow_mut().get_feedback(self.acc_err);
        };
    }
}

impl <_Self: ConnectsFrom, M: LearningRateManager, O: ConnectsFrom> BackProp for HiddenNode <InputNode<_Self>,M,O>{
    fn send_feedback(&mut self){
        for i in &mut self.inputs {
            (**i).borrow_mut().get_feedback(self.acc_err);
        };
    }
}

impl <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> ConnectsTo for HiddenNode <I,M,O>{
    type N = O;
    
    fn connect_to(&mut self, connection: SharedRef<Connection<Self, Self::N>>){
        self.outputs.push(connection);
    }
}

impl <I: ConnectsTo, M: LearningRateManager, O: ConnectsFrom> ConnectsFrom for HiddenNode <I,M,O>{
    type N = I;
    
    fn connect_from(&mut self, connection: SharedRef<Connection<Self::N, Self>>){
        self.inputs.push(connection);
    }
}