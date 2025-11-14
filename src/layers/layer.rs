use crate::{
    learning_rate::{
        LearningRateManager,
        Random,
    },
    SharedRef,
    MultiRef,
    new_shared_ref,
    conn::connection::{
        ConnectsFrom,
        ConnectsTo,
        Connection,
    },
};

fn link<T: ConnectsFrom<N = F>, F: ConnectsTo<N = T>, L: LearningRateManager, I: Layer<F,L>, O: Layer<T, L>, R: Random>(random: &mut R, from: &mut I, to: &mut O) {
    for host in to.neurons() {
        for guest in from.neurons() {
            let conn = Connection::new(guest.clone(), host.clone(), random.rand_float());
            let conn_ref = new_shared_ref(conn);
                (**guest).borrow_mut().connect_to(conn_ref.clone());
                (**host).borrow_mut().connect_from(conn_ref.clone());
        }
    }
}

pub trait Layer <N, L: LearningRateManager>{
    fn neurons(&self) -> &Vec<MultiRef<N>>;
    fn fresh(neuron_ammount: i32,learning_manager: SharedRef<L>) -> Self;
    fn pass_info(&mut self);
}