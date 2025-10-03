use super::input::*;
use crate::conn::{
    connection::Connection,
    connect::Connect,
};

pub trait Out{
    fn activation(&self, x: f32) -> f32;
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid_derivative(x: f32) -> f32 {
    let neg_exp = (-x).exp();
    let base = 1.0 + neg_exp;
    neg_exp / (base * base)
}

pub struct OutputNode<N: Out + Connect> {
    bias: f32,
    val: f32,
    connections: Vec<Connection<N, Self>>
}

impl <N: Out + Connect> Out for OutputNode<N> {
    fn activation(&self, x: f32) -> f32 {
        sigmoid(x - self.bias)
    }
}

impl <N: Out + Connect> In for OutputNode<N>  {
    fn recieve(&mut self, x: f32) {
        self.val += x
    }
}