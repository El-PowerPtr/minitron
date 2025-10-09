use crate::{
    nodes::{
        input::In,
        output::Out,
        learn::*,
    },
    learning_rate::LearningRateManager,
    SharedRef,
    MultiRef,
};

use super::layer::*;

struct HiddenLayer<N: In + Out + Learn + BackProp, M: LearningRateManager>{
    learning_manager: SharedRef<M>,
    nodes: Vec<MultiRef<N>>,
}

impl <N: In + Out + Learn + BackProp, M: LearningRateManager> HiddenLayer<N,M>{
    fn learn(&mut self){
        for i in &mut self.nodes {
            i.borrow_mut().learn();
            i.borrow_mut().send_feedback();
        }
    }
}