use ndarray::{Array, Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub mod galois;
use galois::GF;

/// Compute lagrange coefficients for interpolating a polynomial defined by evaluations at `cpos`
/// to evaluations at `npos`.
/// Returns a matrix with npos.len() rows and cpos.len() columns.
pub fn lagrange_coeffs<const W: u8>(cpos: &[GF<W>], npos: &[GF<W>]) -> Array2<GF<W>> {
    #[cfg(debug_assertions)]
    {
        let num_unique = |v: &[GF<W>]| {
            let mut v: Vec<u32> = v.iter().map(|&x| x.into()).collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        };

        // cpos and npos should not have any repetition.
        assert_eq!(num_unique(cpos), cpos.len());
        assert_eq!(num_unique(npos), npos.len());
    }

    // Pre-compute denominators for all lagrange co-efficients.
    // denom[i] = \prod_{j \neq i} (cpos[i] - cpos[j]).
    let mut denom = Vec::with_capacity(cpos.len());
    cpos.par_iter()
        .map(|v| {
            cpos.par_iter()
                .map(|x| if v == x { GF::ONE } else { v - x })
                .reduce(|| GF::ONE, |acc, x| acc * x)
        })
        .collect_into_vec(&mut denom);

    // Compute lagrange coefficients.
    let coeffs = npos
        .par_iter()
        .flat_map(|v| match cpos.par_iter().position_any(|x| x == v) {
            Some(i) => {
                // If v is a common value between npos and cpos then the lagrange coefficients
                // simplify to selection.
                let mut row = vec![GF::ZERO; cpos.len()];
                row[i] = GF::ONE;
                row
            }
            None => {
                // Pre-compute the numerator = \prod (v - cpos[i]).
                let numerator: GF<W> = cpos
                    .par_iter()
                    .map(|x| v - x)
                    .reduce(|| GF::ONE, |acc, x| acc * x);

                // The lagrange coefficient can now be computed using the pre-computed numerator
                // and denominator.
                cpos.par_iter()
                    .zip_eq(denom.par_iter())
                    .map(|(x, d)| numerator / ((v - x) * d))
                    .collect()
            }
        })
        .collect();

    Array2::from_shape_vec((npos.len(), cpos.len()), coeffs).unwrap()
}

/// Outputs a super-invertible matrix with num_out rows and num_inp columns.
pub fn super_inv_matrix<const W: u8>(num_inp: usize, num_out: usize) -> Array2<GF<W>> {
    debug_assert!(num_inp >= num_out);
    rs_gen_mat(num_out, num_inp).reversed_axes()
}

/// Reads and constructs a binary super invertible matrix from a file.
pub fn binary_super_inv_matrix<const W: u8>(path: &Path) -> Array2<GF<W>> {
    let file = File::open(path).expect(
        "Binary super-invertible matrix should be created using scripts/gen_binary_supmat.py.",
    );
    let reader = BufReader::new(file);

    let mut matrix = Vec::new();
    let mut num_rows = 0;
    for line in reader.lines() {
        let line = line.unwrap();

        for val in line.split(' ') {
            match val {
                "0" => matrix.push(GF::ZERO),
                "1" => matrix.push(GF::ONE),
                _ => panic!("Binary super invertible matrix should only have binary entries"),
            }
        }

        num_rows += 1;
    }

    let num_cols = matrix.len() / num_rows;
    Array::from_shape_vec((num_rows, num_cols), matrix)
        .unwrap()
        .reversed_axes()
}

/// Outputs a reed-solomon generator with code_len rows and mssg_len columns.
pub fn rs_gen_mat<const W: u8>(mssg_len: usize, code_len: usize) -> Array2<GF<W>> {
    let mut matrix = Array::from_elem((code_len, 0), GF::ZERO);

    let col = Array::from_vec(
        (1u32..(code_len + 1).try_into().unwrap())
            .map(GF::from)
            .collect(),
    );
    matrix.push_column(col.view()).unwrap();

    for i in 0..(mssg_len - 1) {
        let res = &col * &matrix.index_axis(ndarray::Axis(1), i);
        matrix.push_column(res.view()).unwrap();
    }

    matrix
}

/// Ordered selections with repititions.
#[derive(Clone)]
pub struct Combination(Vec<usize>);

impl Combination {
    /// Define the selection by a map such that when applied on input `inp` the selection output
    /// `out` satisfies out[i] = inp[map[i]] for each i in len(map).
    /// Such a map can only be applied on inputs whose length is greater than maximum value in the
    /// map.
    pub fn new(map: Vec<usize>) -> Self {
        Self(map)
    }

    /// Infers the mapping from an example.
    /// It is important that the number of unique elements in `inp` is not less than those in
    /// `out`.
    pub fn from_instance<T: Hash + Eq>(inp: &[T], out: &[T]) -> Self {
        let mut lookup = HashMap::new();
        for (i, v) in inp.iter().enumerate() {
            lookup.insert(v, i);
        }

        Self(out.iter().map(|v| *lookup.get(v).unwrap()).collect())
    }

    /// Length of the output.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Apply the combination on a given input.
    pub fn apply<T: Copy>(&self, v: ArrayView1<T>) -> Vec<T> {
        self.0.iter().map(|&i| v[i]).collect()
    }
}
