use crate::{
    conn::connection::ConnectsTo,
    nodes::{
        node::Node,
        output::Out,
        learn::Learn,
        output_node::OutputNode,
    },
    learning_rate::{
        LearningRateManager,
        Random,
    },
    compare_floats,
    MultiRef,
    SharedRef,
    new_multi_ref,
};
use super::layer::Layer;

pub struct OutputLayer <I: ConnectsTo + Learn, M: LearningRateManager + Random, T: Clone> {
    nodes:Vec< MultiRef<OutputNode<I,M,T>>>,
    learning_manager: SharedRef<M>,
}

impl <I: ConnectsTo + Learn, M: LearningRateManager + Random, T: Clone> Layer<OutputNode<I,M,T>, M> for OutputLayer <I,M,T> {

    #[inline]
    fn neurons(&self) -> &Vec<MultiRef<OutputNode<I,M,T>>>{
        &self.nodes
    }
    
    fn fresh(neuron_ammount: i32,learning_manager: SharedRef<M>) -> Self{
        let mut this = OutputLayer {
            learning_manager,
            nodes: vec![],
        };
        for _ in 0..neuron_ammount {
            let node = OutputNode::new(this.learning_manager.borrow_mut().rand_float(), this.learning_manager.clone());
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

impl  <I: ConnectsTo + Learn, M: LearningRateManager + Random, T: Clone> OutputLayer <I,M,T> {
    pub fn winner(&self) -> Option<MultiRef<OutputNode<I,M,T>>> {
        self.nodes.iter()
                    .max_by(|x, y| compare_floats(&(**x).borrow().val(), &(**y).borrow().val()))
                    .cloned()
    }
}