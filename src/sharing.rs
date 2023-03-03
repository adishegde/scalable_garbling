use super::galois::GFElement;

// Compute lagrange coefficients for interpolating a polynomial defined by evaluations at `cpos`
// to evaluations at `npos`.
pub fn lagrange_coeffs(cpos: &[GFElement], npos: &[GFElement]) -> Vec<Vec<GFElement>> {
    let gf_one = GFElement::from(1, npos[0].get_field());
    let gf_zero = GFElement::from(0, npos[0].get_field());

    // Pre-compute denominators for all lagrange co-efficients.
    // denom[i] = \prod_{j \neq i} (cpos[i] - cpos[j]).
    let mut denom = Vec::new();
    for i in 0..cpos.len() {
        denom.push(cpos.iter().enumerate().fold(gf_one, |acc, (idx, &x)| {
            if idx == i {
                acc
            } else {
                acc * (cpos[i] - x)
            }
        }));
    }

    let mut all_coeffs = Vec::new();

    for &v in npos {
        // Compute L_1(v), ..., L_n(v) where L_i is the i-th lagrange polynomial.
        let mut coeffs = Vec::new();

        // We first compute the numerator of L_1(v) i.e., (v - cpos[1]) * ... * (v - cpos.last()).
        let mut numerator = cpos.iter().skip(1).fold(gf_one, |acc, &x| acc * (v - x));
        coeffs.push(numerator / denom[0]);

        for j in 1..cpos.len() {
            // Compute L_j(v).
            if numerator != gf_zero {
                numerator *= (v - cpos[j - 1]) / (v - cpos[j]);
                coeffs.push(numerator / denom[j]);
            } else if v == cpos[j] {
                coeffs.push(gf_one);
            } else {
                coeffs.push(gf_zero);
            }
        }

        all_coeffs.push(coeffs);
    }

    all_coeffs
}
