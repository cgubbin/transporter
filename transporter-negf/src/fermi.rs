// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Fermi
//! Definition of Fermi integrals and their inverses
//! 
//! This module provides fast methods to compute Fermi integrals and inverse Fermi integrals
//! commonly utilised in the calculation of the electronic density, and in solving Poisson's
//! equation. Rather than calculating Fermi integrals by direct numerical integration we follow
//! a [minimax approximation](https://doi.org/10.1016/j.amc.2015.03.015) which allows the values
//! to be computed to a high degree of accuracy from a single lookup.

use nalgebra::RealField;

/// # Inverse Fermi integral of order 0.5.
///
/// The inverse of the Fermi integral of order 0.5, given by
/// $$F\left(\mu\right) = \int_0^{\infty} dx \frac{x^{0.5}}{\exp(x - \mu) +1}$$
/// using a [minimax approximation](https://doi.org/10.1016/j.amc.2015.03.015) to avoid costly
///  numerical integration. For a given value of F this function returns the chemical potential.
#[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
pub(crate) fn inverse_fermi_integral_05<T: Copy + RealField>(mu: T) -> T {
    let mu1 = 5.893_676_232_593_605;
    let mu2 = 20.292_139_527_268_84;
    let mu3 = 69.680_386_701_202_79;
    let mu4 = 246.247_415_252_814_4;
    if mu <= mu1 {
        let v = mu1 - mu;
        let r = mu
            * (4.422_539_954_384_558e9
                + mu * (1.431_882_653_121_693_1e9
                    + mu * (2.002_451_116_208_425_2e8
                        + mu * (1.577_188_595_334_683_7e7
                            + mu * (7.664_073_281_017_675e5
                                + mu * (2.359_936_249_884_790_2e4
                                    + mu * (4.411_460_174_171_255_6e2
                                        + mu * (3.884_445_127_782_172_8))))))))
            / (2.448_084_356_710_615_6e9
                + v * (2.063_907_695_769_060_9e8
                    + v * (6.943_821_586_626_003e6
                        + v * (8.039_397_005_856_418e4
                            + v * (-1.791_261_676_399_435_1e3
                                + v * (-7.908_051_927_048_793e1 - v))))));
        r.ln()
    } else if mu <= mu2 {
        let y = mu - mu1;
        (5.849_893_914_158_47e14
            + y * (3.353_389_340_896_419e14
                + y * (7.300_790_845_633_384e13
                    + y * (7.531_271_098_292_146e12
                        + y * (3.726_221_594_134_586e11
                            + y * (7.827_935_737_269_045e9 + y * (5.021_972_425_404_123e7)))))))
            / (1.439_729_866_842_128e14
                + y * (6.440_007_889_067_505e13
                    + y * (1.049_888_208_290_439_5e13
                        + y * (7.568_424_788_316_453e11
                            + y * (2.320_622_823_577_197_3e10
                                + y * (2.432_978_289_635_439_8e8
                                    + y * (3.676_866_413_386_036e5
                                        + y * (-5.924_317_283_823_515e2 + y))))))))
    } else if mu <= mu3 {
        let y = mu - mu2;
        (6.733_834_344_762_315e18
            + y * (1.138_529_116_708_601_9e18
                + y * (7.441_797_125_810_403e16
                    + y * (2.355_652_759_572_274e15
                        + y * (3.690_410_771_111_407e13
                            + y * (2.592_735_705_594_059_4e11 + y * (5.989_403_440_741_098e8)))))))
            / (6.968_777_783_221_498e17
                + y * (9.451_599_633_557_072e16
                    + y * (4.738_875_908_308_96e15
                        + y * (1.076_651_021_592_855e14
                            + y * (1.088_853_987_040_025_6e12
                                + y * (4.037_404_739_026_029_6e9
                                    + y * (2.312_681_435_753_184e6
                                        + y * (-1.478_829_470_377_447e3 + y))))))))
    } else if mu <= mu4 {
        let y = mu - mu3;
        (7.884_494_095_314_25e19
            + y * (3.748_646_557_381_002e18
                + y * (6.934_193_474_730_825e16
                    + y * (6.302_949_477_641_709e14
                        + y * (2.929_931_660_905_170_4e12
                            + y * (6.591_658_047_866_512e9
                                + y * (6.082_995_857_672_390_5e6
                                    + y * (1.505_484_342_090_580_9e3))))))))
            / (3.559_324_730_480_472e18
                + y * (1.350_579_770_030_645_1e17
                    + y * (1.916_091_921_255_301_8e15
                        + y * (1.265_256_065_109_532_8e13
                            + y * (3.949_105_503_321_385e10
                                + y * (5.253_083_775_042_777e7
                                    + y * (2.225_254_116_592_023_6e4 + y)))))))
    } else {
        let vc = 1_543.460_693_945_940_2;
        let t = vc * mu.powf(-4. / 3.);
        let s = 1. - t;
        let r = (3.433_012_505_914_283_5e7
            + s * (8.713_462_091_032_494e5 + s * (2.424_556_014_825_642e3 + s)))
            / (t * (1.296_167_759_591_953_3e4
                + s * (3.209_288_389_279_310_5e2 + s * (7.192_193_760_323_717e-1))));
        r.sqrt()
    }
}

/// # Fermi integral of order -0.5.
///
/// The Fermi integral of order -0.5, given by
/// $$F\left(\mu\right) = \int_0^{\infty} dx \frac{x^{-0.5}}{\exp(x - \mu) +1}$$
/// using a [minimax approximation](https://doi.org/10.1016/j.amc.2015.03.015) to avoid costly
///  numerical integration.
#[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
pub(crate) fn fermi_integral_m05<T: Copy + RealField>(mu: T) -> T {
    if mu <= -2. {
        let ex = mu.exp();
        let t = ex * 7.389_056_098_930_65;
        ex * (1.772_453_850_905_516
            - ex * (4.064_145_375_102_844e4
                + t * (9.395_708_094_084_644e3
                    + t * (6.499_616_831_526_73e2
                        + t * (1.279_722_958_047_589_6e1 + t * 1.538_643_507_675_854_6e-3))))
                / (3.242_718_847_652_929_3e4
                    + t * (1.107_992_056_612_748e4
                        + t * (1.322_966_270_014_788_5e3 + t * (6.373_836_102_933_347e1 + t)))))
    } else if mu <= 0. {
        let s = -0.5 * mu;
        let t = 1. - s;
        (2.727_700_921_319_327e2
            + t * (3.088_456_538_446_828_5e1
                + t * (-6.435_376_323_803_661
                    + t * (1.487_474_730_982_178_7e1
                        + t * (4.869_288_628_421_426
                            + t * (-1.532_658_345_506_736_6
                                + t * (-1.026_988_983_155_974_9
                                    + t * (-1.776_868_209_286_059_4e-1
                                        - t * 3.771_413_255_092_464_4e-3))))))))
            / (2.930_753_781_876_678_7e2
                + s * (3.058_181_626_862_708e2
                    + s * (2.999_623_954_492_976e2
                        + s * (2.076_408_340_874_942_6e2
                            + s * (9.203_848_031_818_518e1
                                + s * (3.701_649_141_127_912e1
                                    + s * (7.885_009_502_714_205_5 + s)))))))
    } else if mu <= 2. {
        let t = 0.5 * mu;
        (3.531_503_605_682_430_6e3
            + t * (6.077_533_965_842_003e3
                + t * (6.199_770_043_398_133e3
                    + t * (4.412_787_019_195_676e3
                        + t * (2.252_273_430_928_109e3
                            + t * (8.118_409_864_922_409e2
                                + t * (1.918_364_010_536_371_3e2
                                    + t * 2.328_818_389_591_838e1)))))))
            / (3.293_837_025_847_962_7e3
                + t * (1.528_974_740_297_891e3
                    + t * (2.568_485_628_149_860_5e3
                        + t * (9.256_426_465_355_582e2
                            + t * (5.742_324_835_403_599e2
                                + t * (1.328_038_593_206_672_7e2
                                    + t * (2.984_471_665_521_021_3e1 + t)))))))
    } else if mu <= 5. {
        let t = (mu - 2.) / 3.;
        (4.060_707_534_041_182_5e3
            + t * (1.081_272_913_330_527_6e4
                + t * (1.389_756_494_822_425_9e4
                    + t * (1.062_847_498_527_400_2e4
                        + t * (5.107_706_701_906_79e3
                            + t * (1.540_843_301_260_033_7e3
                                + t * (2.844_527_201_129_703e2
                                    + t * 2.952_144_173_584_841_6e1)))))))
            / (1.564_581_956_126_335_4e3
                + t * (2.825_751_722_778_504_2e3
                    + t * (3.189_160_661_699_815_6e3
                        + t * (1.955_039_790_690_325_6e3
                            + t * (8.280_003_336_918_147e2
                                + t * (1.814_981_110_895_183_7e2
                                    + t * (3.203_528_577_948_037_5e1 + t)))))))
    } else if mu <= 10. {
        let t = 0.2 * mu - 1.;
        (1.198_417_190_295_575e3
            + t * (3.263_514_545_549_086_5e3
                + t * (3.874_975_884_713_765e3
                    + t * (2.623_130_603_171_998_2e3
                        + t * (1.100_413_556_371_212_3e3
                            + t * (2.674_695_324_905_036e2
                                + t * (2.542_076_718_127_183_5e1
                                    + t * 3.898_877_542_345_558e-1)))))))
            / (2.734_079_577_925_57e2
                + t * (5.959_183_189_520_586e2
                    + t * (6.052_024_522_616_608e2
                        + t * (3.431_833_027_356_2e2
                            + t * (1.221_876_220_156_957_2e2
                                + t * (2.090_163_590_798_559_3e1 + t))))))
    } else if mu <= 20. {
        let t = 0.1 * mu - 1.;
        (9.446_001_694_352_377e3
            + t * (3.684_344_484_740_286e4
                + t * (6.371_011_154_199_262e4
                    + t * (6.298_521_973_610_748e4
                        + t * (3.763_452_313_957_009e4
                            + t * (1.281_098_986_278_077_6e4
                                + t * (1.981_568_961_389_209_6e3
                                    + t * 8.149_301_718_976_676e1)))))))
            / (1.500_046_978_101_336_7e3
                + t * (5.086_913_810_527_941e3
                    + t * (7.730_015_937_476_219e3
                        + t * (6.640_833_762_393_606e3
                            + t * (3.338_995_903_008_264e3
                                + t * (8.604_990_438_868_03e2
                                    + t * (7.885_658_241_869_267e1 + t)))))))
    } else if mu <= 40. {
        let t = 0.05 * mu - 1.;
        (2.297_796_578_553_672_2e4
            + t * (1.234_166_168_138_877_8e5
                + t * (2.611_537_651_723_551e5
                    + t * (2.746_188_945_140_958e5
                        + t * (1.497_107_183_899_248_5e5
                            + t * (4.012_933_717_001_846e4
                                + t * (4.470_464_958_814_151e3
                                    + t * 1.326_843_468_310_029_8e2)))))))
            / (2.571_688_425_253_357e3
                + t * (1.252_149_822_907_753_5e4
                    + t * (2.326_815_743_250_553_4e4
                        + t * (2.047_723_201_197_581_5e4
                            + t * (8.726_525_779_622_682e3
                                + t * (1.647_428_968_967_699_1e3
                                    + t * (1.064_752_751_420_766_3e2 + t)))))))
    } else {
        let factor = 2.;
        let w = mu.powi(-2);
        let t = 1600. * w;
        mu.sqrt()
            * factor
            * (1.
                - w * (4.112_335_167_120_099_7e-1
                    + t * (1.109_804_100_340_889_5e-3
                        + t * (1.136_892_989_901_736_8e-5
                            + t * (2.569_317_906_794_368e-7
                                + t * (9.978_977_867_554_462e-9
                                    + t * 8.676_676_987_911_086e-10))))))
    }
}

/// # Fermi integral of order 0.5.
///
/// The Fermi integral of order 0.5, given by
/// $$F\left(\mu\right) = \int_0^{\infty} dx \frac{x^{0.5}}{\exp(x - \mu) +1}$$
/// using a [minimax approximation](https://doi.org/10.1016/j.amc.2015.03.015) to avoid costly
/// numerical integration.
#[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
pub fn fermi_integral_05<T: Copy + RealField>(mu: T) -> T {
    if mu <= -2. {
        let ex = mu.exp();
        let t = ex * 7.389_056_098_930_65;
        ex * (8.862_269_254_527_58e-1
            - ex * (1.989_445_533_869_516_7e4
                + t * (4.509_643_299_559_486e3
                    + t * (3.034_617_890_351_424e2
                        + t * (5.757_487_911_475_474 + t * 2.750_889_868_497_626e-3))))
                / (6.349_391_504_130_805e4
                    + t * (1.907_011_782_436_039_4e4
                        + t * (1.962_193_621_412_351e3 + t * (7.925_070_495_864_016e1 + t)))))
    } else if mu <= 0. {
        let s = -0.5 * mu;
        let t = 1. - s;
        (1.494_625_877_688_652_3e2
            + t * (2.281_258_898_850_501_6e1
                + t * (-6.292_563_955_342_855e-1
                    + t * (9.081_204_415_159_952
                        + t * (3.353_574_784_018_353
                            + t * (-4.736_776_969_155_558e-1
                                + t * (-4.671_909_135_561_859_7e-1
                                    + t * (-8.806_103_172_723_308e-2
                                        - t * 2.622_080_804_915_726_7e-3))))))))
            / (269.94660938022644e0
                + s * (343.6419926336247e0
                    + s * (323.9049470901941e0
                        + s * (218.89170769294024e0
                            + s * (102.31331350098315e0
                                + s * (36.319337289702664e0 + s * (8.331_740_123_138_946 + s)))))))
    } else if mu <= 2. {
        let t = 0.5 * mu;
        (7.165_271_711_921_555e4
            + t * (1.349_547_340_702_237_6e5
                + t * (1.536_938_333_503_156_3e5
                    + t * (1.232_472_807_457_034e5
                        + t * (7.288_629_364_793_072e4
                            + t * (3.208_124_994_223_629_6e4
                                + t * (1.021_099_673_377_629_2e4
                                    + t * (2.152_711_103_813_208e3
                                        + t * 2.329_065_881_652_050_5e2))))))))
            / (1.056_678_398_542_988e5
                + t * (3.194_607_529_893_144_6e4
                    + t * (7.115_878_877_642_221e4
                        + t * (1.565_089_901_381_874_2e4
                            + t * (1.352_180_336_577_834_4e4
                                + t * (1.646_982_582_835_279e3
                                    + t * (6.189_069_196_924_941e2
                                        + t * (-3.363_195_917_553_947 + t))))))))
    } else if mu <= 5. {
        let t = (mu - 2.) / 3.;
        (2.374_487_069_933_143e4
            + t * (6.825_785_898_556_23e4
                + t * (8.932_744_676_833_347e4
                    + t * (6.276_634_156_004_425e4
                        + t * (2.009_366_226_099_02e4
                            + t * (-2.213_890_841_197_779_5e3
                                + t * (-3.901_660_572_675_774e3 - t * 9.486_428_959_448_589e2)))))))
            / (9.488_619_729_195_658e3
                + t * (1.251_481_255_269_530_8e4
                    + t * (9.903_440_882_074_51e3
                        + t * (2.138_154_209_103_343e3
                            + t * (-5.283_948_637_308_382e2
                                + t * (-6.610_336_339_954_497e2
                                    + t * (-5.144_814_702_509_623e1 + t)))))))
    } else if mu <= 10. {
        let t = 0.2 * mu - 1.;
        (3.113_374_526_615_825_6e5
            + t * (1.112_670_744_166_482e6
                + t * (1.756_386_288_956_717_4e6
                    + t * (1.596_308_558_037_724_6e6
                        + t * (9.108_189_354_561_837e5
                            + t * (3.264_927_335_507_012_3e5
                                + t * (6.550_726_249_728_529_4e4
                                    + t * 4.809_456_495_272_869e3)))))))
            / (3.972_166_416_250_897e4
                + t * (8.642_475_291_076_624e4
                    + t * (8.816_372_552_521_518e4
                        + t * (5.061_573_635_111_574e4
                            + t * (1.733_497_748_050_082e4
                                + t * (2.712_131_708_090_425_5e3
                                    + t * (8.222_058_283_546_29e1 - t)))))))
            * 9.999_999_999_999_999e-1
    } else if mu <= 20. {
        let t = 0.1 * mu - 1.;
        (7.268_700_630_030_598e6
            + t * (2.790_497_348_547_760_4e7
                + t * (4.427_917_677_597_424e7
                    + t * (3.637_350_175_123_633_4e7
                        + t * (1.557_663_424_636_798e7
                            + t * (2.974_693_570_852_995e6 + t * 1.545_164_470_315_984e5))))))
            / (3.405_425_443_602_097e5
                + t * (8.050_214_686_476_201e5
                    + t * (7.590_882_354_550_026e5
                        + t * (3.046_866_713_716_403_5e5
                            + t * (3.928_940_614_005_423_4e4
                                + t * (5.824_261_381_263_983e2
                                    + t * (1.127_281_945_815_860_3e1 - t)))))))
    } else if mu <= 40. {
        let t = 0.05 * mu - 1.;
        (4.814_497_975_419_631e6
            + t * (1.851_628_507_131_276e7
                + t * (2.776_309_675_225_744_4e7
                    + t * (2.032_759_376_880_706e7
                        + t * (7.415_788_715_893_693e6
                            + t * (1.211_931_135_961_890_5e6 + t * 6.321_195_451_446_449e4))))))
            / (8.049_277_659_752_374e4
                + t * (1.893_286_781_526_548_4e5
                    + t * (1.511_558_906_514_825_6e5
                        + t * (4.814_632_422_538_372e4
                            + t * (5.407_088_783_941_806e3
                                + t * (1.121_950_444_107_755_8e2 - t))))))
    } else {
        let w = mu.powi(-2);
        let s = 1. - 1600. * w;
        mu * mu.sqrt()
            * 6.666_666_666_666_666e-1
            * (1.
                + w * (8.109_793_907_444_779_5e3
                    + s * (3.420_698_674_547_041e2 + s * 1.071_417_022_935_046))
                    / (6.569_984_725_328_291e3 + s * (2.807_064_658_516_838e2 + s)))
    }
}

#[cfg(test)]
mod test {
    use nalgebra::RealField;

    /// # Inverse Fermi integral of order 0.5.
    ///
    /// The inverse of the Fermi integral of order 0.5, given by
    /// $$F\left(\mu\right) = \int_0^{\infty} dx \frac{x^{0.5}}{\exp(x - \mu) +1}$$
    /// using a [minimax approximation](https://doi.org/10.1016/j.amc.2015.03.015) to avoid costly
    ///  numerical integration.
    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn inverse_fermi_integral_m05<T: Copy + RealField>(mu: T) -> T {
        assert!(mu > 0.);
        let mu1 = 2.344_105_321_212_492_5;
        let mu2 = 3.847_581_046_412_509;
        let mu3 = 5.627_089_693_446_789_6;
        let mu4 = 8.074_771_479_066_275;
        let mu5 = 12.670_348_407_162_363;
        if mu <= mu1 {
            let v = mu1 - mu;
            let r = mu
                * (4.145_586_706_362_006e3
                    + v * (4.649_248_922_882_589e3
                        + v * (2.546_827_947_260_463_6e3
                            + v * (8.261_718_336_151_982e2
                                + v * (1.652_332_434_289_845_7e2
                                    + v * (1.913_909_833_389_933e1
                                        + v * (1.000_023_352_316_797)))))))
                / (1.833_926_093_938_476_5e3
                    + v * (3.659_014_106_636_190_3e3
                        + v * (3.312_582_823_191_93e3
                            + v * (1.766_105_894_662_950_7e3
                                + v * (5.999_587_832_536_569e2
                                    + v * (1.302_243_920_251_478_4e2
                                        + v * (1.679_576_227_495_763_5e1 + v)))))));
            r.ln()
        } else if mu <= mu2 {
            let y = mu - mu1;
            (1.000_564_003_768_906e5
                + y * (9.379_587_690_169_092e4
                    + y * (1.196_843_677_580_723_7e4
                        + y * (3.605_605_662_316_988e3
                            + y * (3.003_239_025_477_387e3
                                + y * (2.066_629_315_269_846_6e2
                                    + y * (4.696_718_836_497_964e1
                                        + y * (1.265_579_992_909_832_7e1))))))))
                / (6.000_423_227_232_110_4e4
                    + y * (9.459_146_043_313_48e3
                        + y * (-3.026_014_341_143_112_2e3
                            + y * (2.205_326_139_949_497_7e3
                                + y * (1.666_464_494_929_515_2e2
                                    + y * (-6.113_590_366_933_374e1
                                        + y * (1.713_748_128_878_325_6e1 - y)))))))
        } else if mu <= mu3 {
            let y = mu - mu2;
            (7.807_708_032_862_033e5
                + y * (7.451_198_020_769_327e5
                    + y * (4.857_013_913_509_063e5
                        + y * (2.101_260_997_620_376e5
                            + y * (6.593_059_012_590_01e4
                                + y * (1.608_233_368_267_812e4
                                    + y * (2.307_050_723_848_604e3
                                        + y * (2.145_957_705_253_772_8e2))))))))
                / (1.976_108_793_787_063_3e5
                    + y * (9.747_386_750_054_675e4
                        + y * (6.569_798_537_322_176e4
                            + y * (1.581_081_503_048_769_6e4
                                + y * (5.143_706_463_450_254e3
                                    + y * (5.459_142_145_070_227e2
                                        + y * (2.546_664_870_149_129_5e1 - y)))))))
        } else if mu <= mu4 {
            let y = mu - mu3;
            (9.723_633_697_100_386e7
                + y * (1.016_454_525_292_165_1e8
                    + y * (5.610_803_657_489_166_4e7
                        + y * (2.053_326_211_341_858_3e7
                            + y * (5.262_211_474_540_098e6
                                + y * (9.659_593_698_445_203e5
                                    + y * (1.121_556_937_485_578_7e5
                                        + y * (6.002_807_590_319_267e3))))))))
                / (1.211_283_183_281_239_7e7
                    + y * (8.485_660_688_557_204e6
                        + y * (3.664_182_665_580_148_3e6
                            + y * (1.019_917_572_918_082_2e6
                                + y * (1.864_575_110_128_001_5e5
                                    + y * (2.335_157_206_436_227e4
                                        + y * (3.637_541_867_888_401e1 - y)))))))
        } else if mu <= mu5 {
            let y = mu - mu4;
            (9.859_217_899_528_706e10
                + y * (9.212_482_052_867_464e10
                    + y * (4.659_030_699_733_764e10
                        + y * (1.500_219_539_387_355_4e10
                            + y * (3.429_550_982_788_585e9
                                + y * (5.388_776_274_849_923e8
                                    + y * (4.856_263_306_490_216_4e7
                                        + y * (1.797_949_664_146_387_7e6))))))))
                / (6.029_507_485_001_504e9
                    + y * (4.150_044_565_021_771_4e9
                        + y * (1.734_761_287_835_878e9
                            + y * (4.265_995_009_656_092_5e8
                                + y * (7.803_780_012_911_421e7
                                    + y * (7.194_457_928_823_352e6
                                        + y * (-7.176_409_041_887_143e1 + y)))))))
        } else {
            let wc = 25_798.742_392_161_876;
            let t = wc / mu.powi(4);
            let s = 1. - t;
            let r = (1.517_032_819_953_948e6
                + s * (1.790_238_358_014_511_2e5 + s * (2.924_085_877_500_064_3e3 + s)))
                / (t * (9.398_803_527_483_199e2
                    + s * (1.118_768_502_037_483_9e2
                        + s * (1.923_902_482_284_060_2 + s * (2.343_662_193_240_92e-3)))));
            r.sqrt()
        }
    }

    #[test]
    fn inverse_fermi_integral_of_order_05_returns_correct_result() {
        let values = (0..100).map(|idx| idx as f64 / 100_f64);
        for value in values {
            let fermi_integral = super::fermi_integral_05(value);
            let inverse_fermi_integral = super::inverse_fermi_integral_05(fermi_integral);
            approx::assert_relative_eq!(inverse_fermi_integral, value, epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse_fermi_integral_of_order_m05_returns_correct_result() {
        let values = (0..100).map(|idx| idx as f64 / 100_f64);
        for value in values {
            let fermi_integral = super::fermi_integral_m05(value);
            let inverse_fermi_integral = inverse_fermi_integral_m05(fermi_integral);
            approx::assert_relative_eq!(inverse_fermi_integral, value, epsilon = 1e-10);
        }
    }

    use rand::Rng;
    #[test]
    fn calculated_fermi_level_gives_correct_electronic_density() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let ratio: f64 = rng.gen(); // The ratio of carrier concentration to intrinsic electron density
            let gamma = std::f64::consts::PI.sqrt() / 2.;
            let fermi_level = super::inverse_fermi_integral_05(gamma * ratio);
            let value = super::fermi_integral_05(fermi_level) / gamma;
            approx::assert_relative_eq!(value, ratio, epsilon = std::f64::EPSILON * 100_f64);
        }
    }

    #[test]
    fn numerical_electron_density_jacobian_matches_analytical() {
        let mut rng = rand::thread_rng();
        let delta = 1e-10;
        let phi: f64 = rng.gen();
        let denominator: f64 = rng.gen();

        for _ in 0..100 {
            let ratio: f64 = rng.gen();
            let dn_dphi = super::fermi_integral_m05((ratio + phi) / denominator) / denominator;
            let n = |phi| super::fermi_integral_05((ratio + phi) / denominator);

            let numerical = (n(phi + delta) - n(phi - delta)) / delta;

            approx::assert_relative_eq!(dn_dphi, numerical, epsilon = 100000_f64 * delta);
        }
    }
}
