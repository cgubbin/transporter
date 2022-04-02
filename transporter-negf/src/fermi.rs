use nalgebra::RealField;

#[allow(clippy::excessive_precision)]
#[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
pub(crate) fn inverse_fermi_integral_p<T: Copy + RealField>(u: T) -> T {
    // Computes the inverse fermi integral of order 0.5 using the minimax
    // approximation outlined in DOI: 10.1016/j.amc.2015.03.015
    //assert!(u > 0.);
    let u1 = 5.8936762325936050502;
    let u2 = 20.292139527268839575;
    let u3 = 69.680386701202787485;
    let u4 = 246.24741525281437791;
    if u <= u1 {
        let v = u1 - u;
        let r = u
            * (4.4225399543845577739e9
                + u * (1.4318826531216930391e9
                    + u * (2.0024511162084252731e8
                        + u * (1.5771885953346837109e7
                            + u * (7.664073281017674960e5
                                + u * (2.3599362498847900809e4
                                    + u * (4.4114601741712557348e2
                                        + u * (3.8844451277821727933e0))))))))
            / (2.448084356710615572e9
                + v * (2.063907695769060888e8
                    + v * (6.943821586626002503e6
                        + v * (8.039397005856418743e4
                            + v * (-1.791261676399435220e3 + v * (-7.908051927048792349e1 - v))))));
        r.ln()
    } else if u <= u2 {
        let y = u - u1;
        (5.849893914158469793e14
            + y * (3.353389340896418967e14
                + y * (7.300790845633384552e13
                    + y * (7.531271098292146768e12
                        + y * (3.726221594134586141e11
                            + y * (7.827935737269045014e9 + y * (5.02197242540412350e7)))))))
            / (1.4397298668421281743e14
                + y * (6.440007889067504875e13
                    + y * (1.0498882082904393876e13
                        + y * (7.568424788316453035e11
                            + y * (2.3206228235771973103e10
                                + y * (2.4329782896354397638e8
                                    + y * (3.6768664133860359837e5
                                        + y * (-5.924317283823514482e2 + y))))))))
    } else if u <= u3 {
        let y = u - u2;
        (6.733834344762314874e18
            + y * (1.1385291167086018856e18
                + y * (7.441797125810403052e16
                    + y * (2.3556527595722738211e15
                        + y * (3.6904107711114070061e13
                            + y * (2.5927357055940595308e11 + y * (5.989403440741097470e8)))))))
            / (6.968777783221497285e17
                + y * (9.451599633557071205e16
                    + y * (4.7388759083089595117e15
                        + y * (1.0766510215928549449e14
                            + y * (1.0888539870400255904e12
                                + y * (4.0374047390260294467e9
                                    + y * (2.3126814357531839818e6
                                        + y * (-1.4788294703774470115e3 + y))))))))
    } else if u <= u4 {
        let y = u - u3;
        (7.884494095314249799e19
            + y * (3.7486465573810023777e18
                + y * (6.934193474730824900e16
                    + y * (6.302949477641708425e14
                        + y * (2.9299316609051704688e12
                            + y * (6.591658047866512380e9
                                + y * (6.082995857672390394e6 + y * (1.5054843420905807932e3))))))))
            / (3.5593247304804720533e18
                + y * (1.3505797700306451874e17
                    + y * (1.9160919212553016350e15
                        + y * (1.2652560651095328402e13
                            + y * (3.9491055033213850687e10
                                + y * (5.253083775042776560e7
                                    + y * (2.2252541165920236251e4 + y)))))))
    } else {
        let vc = 1543.4606939459401869;
        let t = vc * u.powf(-4. / 3.);
        let s = 1. - t;
        let r = (3.4330125059142833612e7
            + s * (8.713462091032493289e5 + s * (2.4245560148256419080e3 + s)))
            / (t * (1.2961677595919532465e4
                + s * (3.2092883892793106635e2 + s * (0.7192193760323717351e0))));
        r.sqrt()
    }
}
