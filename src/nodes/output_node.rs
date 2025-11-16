use super::{
    input::In,
    output::{
        Out,
        sigmoid, 
        sigmoid_derivative,
    },
    learn::{Learn, BackProp},
    label::Label,
};
use crate::{
    nodes::node::{
        Node,
    },
    conn::connection::{
        Connection,
        ConnectsTo,
    },
    learning_rate::LearningRateManager,
    SharedRef,
};

pub struct OutputNode<I: ConnectsTo, M: LearningRateManager,T: Clone> {
    learning_manager: SharedRef<M>,
    err: f32,
    bias: f32,
    val: f32,
    connections: Vec<Connection<I, Self>>,
    label: Option<Label<T>>,
}

impl <I: ConnectsTo, M:LearningRateManager, T: Clone> Out for OutputNode<I,M,T> {
    #[inline]
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x - self.bias)
    }
    #[inline]
    fn local_gradient(&self, err: f32) -> f32 {
        err * sigmoid_derivative(self.val + self.bias)
    }
    
    fn forward_prop(&mut self) {
        if let Some(Label::Training{expected_val,..}) = self.label {
            self.err = expected_val - self.activation(self.val);
        }
    }
}

impl <I: ConnectsTo, M: LearningRateManager, T: Clone> In for OutputNode<I,M,T>  {
    fn recieve(&mut self, x: f32) {
        self.val += x;
    }
}

impl <I: ConnectsTo, M: LearningRateManager, T: Clone> Learn for OutputNode<I,M,T> {
    fn learn(&mut self){
        let local_gradient = self.local_gradient(self.err);
        self.bias -= self.learning_manager.borrow().learning_rate() * local_gradient;
    }
    
    fn get_feedback(&mut self, feedback: f32){
        self.err += feedback;
    }
}

impl <I: ConnectsTo + Learn, M: LearningRateManager, T: Clone> BackProp for OutputNode <I,M,T>{
    fn send_feedback(&mut self){
        for i in &mut self.connections {
            i.get_feedback(self.err);
        };
    }
}

impl <I: ConnectsTo + Learn, M: LearningRateManager, T: Clone> OutputNode<I,M,T> {
    pub fn change_to_usage(&mut self) {
        if let Some(label) = &self.label {
            if let Label::Training{..} = label{
                self.label = Some(label.to_usage());
            }
        } else {
            panic!("Your label is not initialized and you are calling OutputNode::change_to_usage. Fix your mess before pushing to production, nigga!");
        }
    }
    
    pub fn set_label(&mut self, label: Label<T>) {
        self.label = Some(label);
    }
    
    #[inline]
    pub fn val(&self) -> f32 {
        self.val
    }
}

impl <I: ConnectsTo + Learn, M: LearningRateManager, T: Clone> Node<M> for OutputNode<I,M,T> {
    fn new(bias: f32, manager: SharedRef<M>) -> Self {
        OutputNode {
            learning_manager: manager,
            err: 0.0,
            bias,
            val: 0.0,
            connections: vec![],
            label: None,
        }
    }
}