use crate::{
    new_multi_ref,
    nodes::{
        node::Node,
        learn::{Learn, BackProp},
    },
    conn::connection::ConnectsTo,
    learning_rate::{
        LearningRateManager,
        Random,
    },
    SharedRef,
    MultiRef,
};

use super::layer::Layer;

struct HiddenLayer<N: ConnectsTo + Node<M> + Learn + BackProp, M: LearningRateManager + Random>{
    learning_manager: SharedRef<M>,
    nodes: Vec<MultiRef<N>>,
}

impl <N: Node<M> + ConnectsTo + Learn + BackProp, M: LearningRateManager + Random> HiddenLayer<N,M>{
    fn learn(&mut self){
        for i in &mut self.nodes {
            (**i).borrow_mut().learn();
            (**i).borrow_mut().send_feedback();
        }
    }
}

impl <N: ConnectsTo + Node<M> + Learn + BackProp, M: LearningRateManager + Random> Layer<N,M> for HiddenLayer<N,M> {
    fn neurons(&self) -> &Vec<MultiRef<N>>{
        &self.nodes
    }
    
    fn fresh(neuron_ammount: i32,learning_manager: SharedRef<M>) -> Self{
        let mut this = HiddenLayer {
            learning_manager,
            nodes: vec![],
        };
        for _ in 0..neuron_ammount {
            let node = N::new(this.learning_manager.borrow_mut().rand_float(), this.learning_manager.clone());
            let node_ref = new_multi_ref(node);
            this.nodes.push(node_ref);
        }
        this
    }
    fn pass_info(&mut self) {
        for node in &mut self.nodes {
            (**node).borrow_mut().forward_prop();
        }
    }
}