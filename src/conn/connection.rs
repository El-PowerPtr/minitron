use crate::{
    MultiRef,
    nodes::{
        input::In,
        output::Out,
    },
};
use super::connect::Connect;

pub struct Connection <F: Connect + Out, T: In> {
    pub from: MultiRef<F>,
    pub weight: f32,
    pub to: MultiRef<T>,
}