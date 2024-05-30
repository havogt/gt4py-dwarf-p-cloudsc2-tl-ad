import jax

jax.config.update("jax_enable_x64", True)

import gt4py.next as gtx
from gt4py.next.common import Dims
from gt4py.next import broadcast
from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.fbuiltins import exp, sqrt, tanh, minimum, maximum, where


constants = FrozenNamespace(
    **{
        "R2ES": 380.1608703442847,
        "R3LES": 17.502,
        "R3IES": 22.587,
        "R4LES": 32.19,
        "R4IES": -0.7,
        "R5LES": 4217.45694,
        "R5IES": 6185.67582,
        "R5ALVCP": 10497584.68169531,
        "R5ALSCP": 17451123.253362577,
        "RALVDCP": 2489.0792795374246,
        "RALSDCP": 2821.2152982440934,
        "RALFDCP": 332.1360187066693,
        "RTWAT": 273.16,
        "RTICE": 250.16000000000003,
        "RTICECU": 250.16000000000003,
        "RTWAT_RTICE_R": 0.043478260869565216,
        "RTWAT_RTICECU_R": 0.043478260869565216,
        "RKOOP1": 2.583,
        "RKOOP2": 0.0048116,
        "RVTMP2": 0.0,
        "RG": 9.80665,
        "RD": 287.0596736665907,
        "RCPD": 1004.7088578330674,
        "RETV": 0.6077667316114637,
        "RLVTT": 2500800.0,
        "RLSTT": 2834500.0,
        "RLMLT": 333700.0,
        "RTT": 273.16,
        "RV": 461.5249933083879,
        "RAMID": 0.8,
        "RCLDIFF": 3e-06,
        "RCLDIFF_CONVI": 7.0,
        "RCLCRIT": 0.0004,
        "RCLCRIT_SEA": 0.00025,
        "RCLCRIT_LAND": 0.00055,
        "RKCONV": 0.00016666666666666666,
        "RPRC1": 100.0,
        "RPRC2": 0.5,
        "RCLDMAX": 0.005,
        "RPECONS": 5.54725619859993e-05,
        "RVRFACTOR": 0.00509,
        "RPRECRHMAX": 0.7,
        "RTAUMEL": 7200.0,
        "RAMIN": 1e-08,
        "RLMIN": 1e-08,
        "RKOOPTAU": 10800.0,
        "RCLDTOPP": 100.0,
        "RLCRITSNOW": 3e-05,
        "RSNOWLIN1": 0.001,
        "RSNOWLIN2": 0.03,
        "RICEHI1": 3.3333333333333335e-05,
        "RICEHI2": 0.004291845493562232,
        "RICEINIT": 1e-12,
        "RVICE": 0.13,
        "RVRAIN": 4.0,
        "RVSNOW": 1.0,
        "RTHOMO": 235.16000000000003,
        "RCOVPMIN": 0.1,
        "RCCN": 125.0,
        "RNICE": 0.027,
        "RCCNOM": 0.13,
        "RCCNSS": 0.05,
        "RCCNSU": 0.5,
        "RCLDTOPCF": 0.01,
        "RDEPLIQREFRATE": 0.1,
        "RDEPLIQREFDEPTH": 500.0,
        "RCL_KKAac": 67.0,
        "RCL_KKBac": 1.15,
        "RCL_KKAau": 1350.0,
        "RCL_KKBauq": 2.47,
        "RCL_KKBaun": -1.79,
        "RCL_KK_cloud_num_sea": 50.0,
        "RCL_KK_cloud_num_land": 300.0,
        "RCL_AI": 0.069,
        "RCL_BI": 2.0,
        "RCL_CI": 16.8,
        "RCL_DI": 0.527,
        "RCL_X1I": 2000000.0,
        "RCL_X2I": 0.0,
        "RCL_X3I": 1.0,
        "RCL_X4I": 0.0,
        "RCL_CONST1I": 3.6231880115136998e-06,
        "RCL_CONST2I": 6283185.307179586,
        "RCL_CONST3I": 596.9998475835998,
        "RCL_CONST4I": 0.6666666666666666,
        "RCL_CONST5I": 0.9211666666666667,
        "RCL_CONST6I": 1.0000000948961185,
        "RCL_APB1": 714000000000.0,
        "RCL_APB2": 116000000.0,
        "RCL_APB3": 241.6,
        "RCL_AS": 0.069,
        "RCL_BS": 2.0,
        "RCL_CS": 16.8,
        "RCL_DS": 0.527,
        "RCL_X1S": 2000000.0,
        "RCL_X2S": 0.0,
        "RCL_X3S": 1.0,
        "RCL_X4S": 0.0,
        "RCL_CONST1S": 3.6231880115136998e-06,
        "RCL_CONST2S": 6283185.307179586,
        "RCL_CONST3S": 596.9998475835998,
        "RCL_CONST4S": 0.6666666666666666,
        "RCL_CONST5S": 0.9211666666666667,
        "RCL_CONST6S": 1.0000000948961185,
        "RCL_CONST7S": 90363515.76351073,
        "RCL_CONST8S": 1.1756666666666666,
        "RDENSWAT": 1000.0,
        "RDENSREF": 1.0,
        "RCL_AR": 523.5987755982989,
        "RCL_BR": 3.0,
        "RCL_CR": 386.8,
        "RCL_DR": 0.67,
        "RCL_X1R": 0.22,
        "RCL_X2R": 2.2,
        "RCL_X4R": 0.0,
        "RCL_KA273": 0.024,
        "RCL_CDENOM1": 557000000000.0,
        "RCL_CDENOM2": 103000000.0,
        "RCL_CDENOM3": 204.0,
        "RCL_SCHMIDT": 0.6,
        "RCL_DYNVISC": 1.717e-05,
        "RCL_CONST1R": 1.382300767579509,
        "RCL_CONST2R": 2143.2299120517614,
        "RCL_CONST3R": 0.6349999999999998,
        "RCL_CONST4R": -0.20000000000000018,
        "RCL_CONST5R": 8685252.965082133,
        "RCL_CONST6R": -4.8,
        "RCL_FAC1": 4146.902789847063,
        "RCL_FAC2": 0.5555555555555556,
        "RCL_FZRAB": -0.66,
        "RCL_FZRBB": 200.0,
        "LCLDEXTRA": False,
        "LCLDBUDGET": False,
        "NSSOPT": 1,
        "NCLDTOP": 15,
        "NAECLBC": 9,
        "NAECLDU": 4,
        "NAECLOM": 7,
        "NAECLSS": 1,
        "NAECLSU": 11,
        "NCLDDIAG": 0,
        "NAERCLD": 0,
        "LAERLIQAUTOLSP": False,
        "LAERLIQAUTOCP": False,
        "LAERLIQAUTOCPB": False,
        "LAERLIQCOLL": False,
        "LAERICESED": False,
        "LAERICEAUTO": False,
        "NSHAPEP": 2,
        "NSHAPEQ": 2,
        "NBETA": 100,
        "LTLEVOL": False,
        "LPHYLIN": True,
        "LENOPERT": True,
        "LEPPCFLS": False,
        "LRAISANEN": True,
        "RLPTRC": 266.42345596729064,
        "RLPAL1": 0.15,
        "RLPAL2": 20.0,
        "RLPBB": 5.0,
        "RLPCC": 5.0,
        "RLPDD": 5.0,
        "RLPMIXL": 4000.0,
        "RLPBETA": 0.2,
        "RLPDRAG": 0.0,
        "RLPEVAP": 0.0,
        "RLPP00": 30000.0,
        "LREGCL": False,
        "LEVAPLS2": False,
        "ICALL": 0,
        "LDRAIN1D": False,
        "NLEV": 137,
        "ZEPS1": 1e-12,
        "ZEPS2": 1e-10,
        "ZQMAX": 0.5,
        "ZSCAL": 0.9,
    }
)

I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

Koff = gtx.FieldOffset("K", source=K, target=(K,))

IJKField = gtx.Field[Dims[I, J, K], gtx.float64]
IJField = gtx.Field[Dims[I, J], gtx.float64]
KField = gtx.Field[Dims[K], gtx.float64]


@gtx.field_operator
def _compute_t(in_t: IJKField, in_tnd_cml_t: IJKField, dt: gtx.float64) -> IJKField:
    return in_t + dt * in_tnd_cml_t


@gtx.scan_operator(axis=K, forward=True, init=0.1, strategy="jax")
def _compute_trpaus_forward(
    s: gtx.float64, t: gtx.float64, tp1: gtx.float64, in_eta: gtx.float64
) -> gtx.float64:
    return where((in_eta > 0.1) & (in_eta < 0.4) & (t > tp1), in_eta, s)


@gtx.scan_operator(axis=K, forward=False, init=(True, 0.0), strategy="jax")
def _propagate(
    s: tuple[bool, gtx.float64], trpaus: gtx.float64
) -> tuple[bool, gtx.float64]:
    first, val = s
    res = where(first, (False, trpaus), (False, val))
    return res


@gtx.field_operator
def _compute_trpaus(t: IJKField, in_eta: KField) -> IJKField:
    return _propagate(_compute_trpaus_forward(t, t(Koff[1]), in_eta))[1]


@gtx.field_operator
def f_foealfa(t: IJKField) -> IJKField:
    return minimum(
        1.0,
        (
            (maximum(constants.RTICE, minimum(constants.RTWAT, t)) - constants.RTICE)
            * constants.RTWAT_RTICE_R
        )
        ** 2.0,
    )


@gtx.field_operator
def f_foeewm(t: IJKField) -> IJKField:
    return constants.R2ES * (
        f_foealfa(t)
        * exp(constants.R3LES * (t - constants.RTT) / (t - constants.R4LES))
        + (1.0 - f_foealfa(t))
        * (exp(constants.R3IES * (t - constants.RTT) / (t - constants.R4IES)))
    )


@gtx.field_operator
def _crh2_eta_ge_trpaus(
    tmp_trpaus: IJKField, in_eta: KField, rh3: gtx.float64
) -> IJKField:
    rh1 = 1.0
    rh2 = (
        0.35
        + 0.14 * ((tmp_trpaus - 0.25) / 0.15) ** 2.0
        + 0.04 * minimum(tmp_trpaus - 0.25, 0.0) / 0.15
    )
    deta2 = 0.3
    bound1 = tmp_trpaus + deta2
    deta1 = 0.09 + 0.16 * (0.4 - tmp_trpaus) / 0.3
    bound2 = 1.0 - deta1

    return where(
        in_eta < bound1,
        rh3 + (rh2 - rh3) * (in_eta - tmp_trpaus) / deta2,
        where(
            in_eta < bound2,
            rh2,
            rh1 + (rh2 - rh1) * sqrt((1.0 - in_eta) / deta1),
        ),
    )


@gtx.field_operator
def _compute_crh2(tmp_trpaus: IJKField, in_eta: KField) -> IJKField:
    # set up critical value of humidity
    rh3 = 1.0
    return where(in_eta < tmp_trpaus, rh3, _crh2_eta_ge_trpaus(tmp_trpaus, in_eta, rh3))


@gtx.field_operator
def _compute_qc_clc_else(
    qsat: IJKField, qt: IJKField, qcrit: IJKField, scalm: KField
) -> tuple[IJKField, IJKField]:
    qpd = qsat - qt
    qcd = qsat - qcrit
    out_clc = 1.0 - sqrt(qpd / (qcd - scalm * (qt - qcrit)))
    qc = (scalm * qpd + (1.0 - scalm) * qcd) * (out_clc**2.0)
    return out_clc, qc


@gtx.field_operator
def _compute_qc_clc(
    q: IJKField,
    ql: IJKField,
    qi: IJKField,
    qcrit: IJKField,
    qsat: IJKField,
    scalm: KField,
) -> tuple[IJKField, IJKField]:
    qt = q + ql + qi

    out_clc, qc = where(
        qt < qcrit,
        (0.0, 0.0),
        where(
            qt >= qsat,
            (1.0, (1.0 - scalm) * (qsat - qcrit)),
            _compute_qc_clc_else(qsat, qt, qcrit, scalm),
        ),
    )

    return out_clc, qc


@gtx.field_operator
def f_cuadjtqs_nl_0(
    ap: gtx.float64,
    t: gtx.float64,
    q: gtx.float64,
    z3es: gtx.float64,
    z4es: gtx.float64,
    z5alcp: gtx.float64,
    zaldcp: gtx.float64,
):
    foeew = constants.R2ES * exp(z3es * (t - constants.RTT) / (t - z4es))
    qsat = minimum(foeew / ap, constants.ZQMAX)
    cor = 1.0 / (1.0 - constants.RETV * qsat)
    qsat = qsat * cor
    z2s = z5alcp / (t - z4es) ** 2.0
    cond = (q - qsat) / (1.0 + qsat * cor * z2s)
    t = t + zaldcp * cond
    q = q - cond
    return t, q


@gtx.field_operator
def f_cuadjtqs_nl(ap: gtx.float64, t: gtx.float64, q: gtx.float64):
    z3es = where(t > constants.RTT, constants.R3LES, constants.R3IES)
    z4es = where(t > constants.RTT, constants.R4LES, constants.R4IES)
    z5alcp = where(t > constants.RTT, constants.R5ALVCP, constants.R5ALSCP)
    zaldcp = where(t > constants.RTT, constants.RALVDCP, constants.RALSDCP)

    # if constants.ICALL == 0:
    t, q = f_cuadjtqs_nl_0(ap, t, q, z3es, z4es, z5alcp, zaldcp)
    t, q = f_cuadjtqs_nl_0(ap, t, q, z3es, z4es, z5alcp, zaldcp)

    return t, q


@gtx.field_operator
def _preciptation_evaporation(
    rfln: gtx.float64,
    sfln: gtx.float64,
    covpclr: gtx.float64,
    covptot_km1: gtx.float64,
    in_qsat: gtx.float64,
    qlim: gtx.float64,
    out_clc: gtx.float64,
    in_ap: gtx.float64,
    tmp_aph_s: gtx.float64,
    dt: gtx.float64,
    corqs: gtx.float64,
    in_aph_p1: gtx.float64,
    in_aph: gtx.float64,
):
    prtot = rfln + sfln
    preclr = prtot * covpclr / covptot_km1

    # this is the humidity in the moisest zcovpclr region
    qe = in_qsat - (in_qsat - qlim) * covpclr / ((1.0 - out_clc) ** 2.0)
    beta = (
        constants.RG
        * constants.RPECONS
        * (sqrt(in_ap / tmp_aph_s) / 0.00509 * preclr / covpclr) ** 0.5777
    )

    # implicit solution
    b = dt * beta * (in_qsat - qe) / (1.0 + dt * beta * corqs)

    dtgdp = dt * constants.RG / (in_aph_p1 - in_aph)
    dpr = minimum(covpclr * b / dtgdp, preclr)
    preclr = preclr - dpr
    covptot_km1 = where(preclr <= 0.0, out_clc, covptot_km1)
    out_covptot = covptot_km1
    # warm proportion
    evapr = dpr * rfln / prtot

    # ice proportion
    evaps = dpr * sfln / prtot

    # TODO: foast type deduction fails here
    return where(
        (prtot > constants.ZEPS2)
        & (covpclr > constants.ZEPS2)
        & (constants.LEVAPLS2 | constants.LDRAIN1D),
        (evapr, evaps, out_covptot),
        (0.0, 0.0, 0.0),
    )


@gtx.scan_operator(
    axis=K,
    forward=True,
    init=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    strategy="jax",
)
def main_scan(
    carry: tuple[
        gtx.float64,
        gtx.float64,
        gtx.float64,
        gtx.float64,
        gtx.float64,
        gtx.float64,
        gtx.float64,
        gtx.float64,
    ],
    out_clc: gtx.float64,
    t: gtx.float64,
    cons2: gtx.float64,
    dp: gtx.float64,
    lfdcp: gtx.float64,
    meltp2: gtx.float64,
    in_qsat: gtx.float64,
    qlim: gtx.float64,
    in_ap: gtx.float64,
    tmp_aph_s: gtx.float64,
    dt: gtx.float64,
    in_aph: gtx.float64,
    in_aph_p1: gtx.float64,
    corqs: gtx.float64,
    condl: gtx.float64,
    condi: gtx.float64,
    lvdcp: gtx.float64,
    lsdcp: gtx.float64,
    in_lude: gtx.float64,
    fwat: gtx.float64,
    gdp: gtx.float64,
    q: gtx.float64,
    dr: gtx.float64,
    rfreeze: gtx.float64,
    fwatr: gtx.float64,
):
    rfl_km1, sfl_km1, covptot_km1, _, _, _, _, _ = carry

    # calculate precipitation overlap
    # simple form based on Maximum Overlap
    covptot_km1 = maximum(covptot_km1, out_clc)
    covpclr = maximum(covptot_km1 - out_clc, 0.0)

    # melting of incoming snow
    cons = cons2 * dp / lfdcp
    snmlt = where(
        sfl_km1 != 0.0, minimum(sfl_km1, cons * maximum(t - meltp2, 0.0)), 0.0
    )

    rfln = rfl_km1 + snmlt
    sfln = sfl_km1 - snmlt

    t = t - snmlt / cons

    rfln = rfln + fwatr * dr
    sfln = sfln + (1.0 - fwatr) * dr

    # precipitation evaporation

    evapr, evaps, out_covptot = _preciptation_evaporation(
        rfln,
        sfln,
        covpclr,
        covptot_km1,
        in_qsat,
        qlim,
        out_clc,
        in_ap,
        tmp_aph_s,
        dt,
        corqs,
        in_aph_p1,
        in_aph,
    )

    rfln = rfln - evapr
    sfln = sfln - evaps

    # update of T and Q tendencies due to:
    # - condensation/evaporation of cloud water/ice
    # - detrainment of convective cloud condensate
    # - evaporation of precipitation
    # - freezing of rain (impact on T only)
    dqdt = -(condl + condi) + (in_lude + evapr + evaps) * gdp
    dtdt = (
        lvdcp * condl
        + lsdcp * condi
        - (
            lvdcp * evapr
            + lsdcp * evaps
            + in_lude * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
            - (lsdcp - lvdcp) * rfreeze
        )
        * gdp
    )

    # first guess T and Q
    t = t + dt * dtdt
    q = q + dt * dqdt
    qold = q

    # clipping of final qv
    t, q = f_cuadjtqs_nl(in_ap, t, q)

    # update rain fraction and freezing
    dq = maximum(qold - q, 0.0)
    dr2 = cons2 * dp * dq
    rfreeze2 = where(t < constants.RTT, fwat * dr2, 0.0)
    fwatr = where(t < constants.RTT, 0.0, 1.0)
    rn = fwatr * dr2
    sn = (1.0 - fwatr) * dr2
    condl = condl + fwatr * dq / dt
    condi = condi + (1.0 - fwatr) * dq / dt
    rfln = rfln + rn
    sfln = sfln + sn
    rfreeze = rfreeze + rfreeze2

    # calculate output tendencies
    out_tnd_q = -(condl + condi) + (in_lude + evapr + evaps) * gdp
    out_tnd_t = (
        lvdcp * condl
        + lsdcp * condi
        - (
            lvdcp * evapr
            + lsdcp * evaps
            + in_lude * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
            - (lsdcp - lvdcp) * rfreeze
        )
        * gdp
    )

    return (
        rfln,
        sfln,
        covptot_km1,
        out_tnd_q,
        out_tnd_t,
        out_covptot,
        rfl_km1,
        sfl_km1,
    )


@gtx.field_operator
def _diag_calc_snow_from_cloud_ice(
    out_clc: IJKField, qiwc: IJKField, t: IJKField, ckcodti: gtx.float64
) -> IJKField:
    # diagnostic calculation of snow production from cloud ice
    icrit = (
        0.0001 if constants.LEVAPLS2 | constants.LDRAIN1D else 2.0 * constants.RCLCRIT
    )
    cldi = qiwc / out_clc
    di = (
        ckcodti
        * exp(0.025 * (t - constants.RTT))
        * (1.0 - exp(-((cldi / icrit) ** 2.0)))
    )
    prs = qiwc - out_clc * cldi * exp(-di)
    return where(out_clc > constants.ZEPS2, prs, 0.0)


@gtx.field_operator
def _diag_calc_rain_from_liquid_water(
    out_clc: IJKField, qlwc: IJKField, ckcodtl: gtx.float64
) -> IJKField:
    # diagnostic calculation of rain production from cloud liquid water
    lcrit = (
        1.9 * constants.RCLCRIT
        if constants.LEVAPLS2 | constants.LDRAIN1D
        else 2.0 * constants.RCLCRIT
    )
    cldl = qlwc / out_clc
    dl = ckcodtl * (1.0 - exp(-((cldl / lcrit) ** 2.0)))
    prr = qlwc - out_clc * cldl * exp(-dl)
    return where(out_clc > constants.ZEPS2, prr, 0.0)


@gtx.field_operator
def _cloudsc2_next(
    in_ap: IJKField,
    in_aph: IJKField,
    in_eta: KField,
    in_lu: IJKField,
    in_lude: IJKField,
    in_mfd: IJKField,
    in_mfu: IJKField,
    in_q: IJKField,
    in_qi: IJKField,
    in_ql: IJKField,
    in_qsat: IJKField,
    in_supsat: IJKField,
    in_t: IJKField,
    in_tnd_cml_q: IJKField,
    in_tnd_cml_qi: IJKField,
    in_tnd_cml_ql: IJKField,
    in_tnd_cml_t: IJKField,
    tmp_aph_s: IJField,
    dt: gtx.float64,
) -> tuple[
    IJKField,
    IJKField,
    IJKField,
    IJKField,
    IJKField,
    IJKField,
    IJKField,
    IJKField,
    IJKField,
    IJKField,
]:
    t = _compute_t(in_t, in_tnd_cml_t, dt)
    trpaus = _compute_trpaus(t, in_eta)

    # first guess values for q, ql and qi
    q = in_q + dt * in_tnd_cml_q + in_supsat
    ql = in_ql + dt * in_tnd_cml_ql
    qi = in_qi + dt * in_tnd_cml_qi

    # set up constants required
    ckcodtl = 2.0 * constants.RKCONV * dt
    ckcodti = 5.0 * constants.RKCONV * dt
    cons2 = 1.0 / (constants.RG * dt)
    cons3 = constants.RLVTT / constants.RCPD
    meltp2 = constants.RTT + 2.0

    # parameter for cloud formation
    scalm = constants.ZSCAL * maximum(in_eta - 0.2, constants.ZEPS1) ** 0.2

    # thermodynamic constants
    dp = in_aph(Koff[1]) - in_aph
    zz = constants.RCPD + constants.RCPD * constants.RVTMP2 * q
    lfdcp = constants.RLMLT / zz
    lsdcp = constants.RLSTT / zz
    lvdcp = constants.RLVTT / zz

    # clear cloud and freezing arrays
    out_clc = 0.0  # TODO delete
    out_covptot = 0.0

    # calculate dqs/dT correction factor
    if constants.LPHYLIN | constants.LDRAIN1D:
        fwat = where(
            t < constants.RTT, 0.545 * (tanh(0.17 * (t - constants.RLPTRC)) + 1.0), 1.0
        )
        z3es = where(t < constants.RTT, constants.R3IES, constants.R3LES)
        z4es = where(t < constants.RTT, constants.R4IES, constants.R4LES)
        foeew = constants.R2ES * exp(z3es * (t - constants.RTT) / (t - z4es))
        esdp = minimum(foeew / in_ap, constants.ZQMAX)
    else:
        fwat = f_foealfa(t)
        foeew = f_foeewm(t)
        esdp = foeew / in_ap
    facw = constants.R5LES / ((t - constants.R4LES) ** 2.0)
    faci = constants.R5IES / ((t - constants.R4IES) ** 2.0)
    fac = fwat * facw + (1.0 - fwat) * faci
    dqsdtemp = fac * in_qsat / (1.0 - constants.RETV * esdp)
    corqs = 1.0 + cons3 * dqsdtemp

    # use clipped state
    qlim = minimum(q, in_qsat)

    crh2 = _compute_crh2(trpaus, in_eta)

    # allow ice supersaturation at cold temperatures
    qsat = where(t < constants.RTICE, in_qsat * (1.8 - 0.003 * t), in_qsat)
    qcrit = crh2 * qsat

    # simple uniform distribution of total water from Leutreut & Li (1990)

    out_clc, qc = _compute_qc_clc(q, ql, qi, qcrit, qsat, scalm)

    # add convective component
    gdp = constants.RG / (in_aph(Koff[1]) - in_aph)
    lude = dt * in_lude * gdp
    lo1 = (lude >= constants.RLMIN) & (in_lu(Koff[1]) >= constants.ZEPS2)

    out_clc = where(
        lo1, out_clc + (1.0 - out_clc) * (1.0 - exp(-lude / in_lu(Koff[1]))), out_clc
    )
    qc = where(lo1, qc + lude, qc)

    # add compensating subsidence component
    rho = in_ap / (constants.RD * t)
    rodqsdp = -rho * in_qsat / (in_ap - constants.RETV * foeew)
    ldcp = fwat * lvdcp + (1.0 - fwat) * lsdcp
    dtdzmo = (
        constants.RG * (1.0 / constants.RCPD - ldcp * rodqsdp) / (1.0 + ldcp * dqsdtemp)
    )
    dqsdz = dqsdtemp * dtdzmo - constants.RG * rodqsdp
    dqc = minimum(dt * dqsdz * (in_mfu + in_mfd) / rho, qc)
    qc = qc - dqc

    # new cloud liquid/ice contents and condensation rates (liquid/ice)
    qlwc = qc * fwat
    qiwc = qc * (1.0 - fwat)
    condl = (qlwc - ql) / dt
    condi = (qiwc - qi) / dt

    prs = _diag_calc_snow_from_cloud_ice(out_clc, qiwc, t, ckcodti)
    qiwc = qiwc - prs

    prr = _diag_calc_rain_from_liquid_water(out_clc, qlwc, ckcodtl)
    qlwc = qlwc - prr

    # new precipitation (rain + snow)
    dr = cons2 * dp * (prr + prs)

    # rain fraction (different from cloud liquid water fraction!)
    rfreeze = where(t < constants.RTT, cons2 * dp * prr, 0.0)
    fwatr = where(t < constants.RTT, 0.0, 1.0)

    (
        _,
        _,
        _,
        out_tnd_q,
        out_tnd_t,
        out_covptot,
        out_fplsl,
        out_fplsn,
    ) = main_scan(
        out_clc,
        t,
        cons2,
        dp,
        lfdcp,
        meltp2,
        in_qsat,
        qlim,
        in_ap,
        tmp_aph_s,
        dt,
        in_aph,
        in_aph(Koff[1]),
        corqs,
        condl,
        condi,
        lvdcp,
        lsdcp,
        in_lude,
        fwat,
        gdp,
        q,
        dr,
        rfreeze,
        fwatr,
    )

    out_fhpsl = -out_fplsl * constants.RLVTT
    out_fhpsn = -out_fplsn * constants.RLSTT

    out_tnd_ql = (qlwc - ql) / dt
    out_tnd_qi = (qiwc - qi) / dt

    return (
        out_clc,
        out_covptot,
        out_fhpsl,
        out_fhpsn,
        out_fplsl,
        out_fplsn,
        out_tnd_q,
        out_tnd_qi,
        out_tnd_ql,
        out_tnd_t,
    )


# @gtx.program  # (backend=gtx.itir_python)
# def cloudsc2_next(
#     in_ap: IJKField,
#     in_aph: IJKField,
#     in_eta: KField,
#     in_lu: IJKField,
#     in_lude: IJKField,
#     in_mfd: IJKField,
#     in_mfu: IJKField,
#     in_q: IJKField,
#     in_qi: IJKField,
#     in_ql: IJKField,
#     in_qsat: IJKField,
#     in_supsat: IJKField,
#     in_t: IJKField,
#     in_tnd_cml_q: IJKField,
#     in_tnd_cml_qi: IJKField,
#     in_tnd_cml_ql: IJKField,
#     in_tnd_cml_t: IJKField,
#     out_clc: IJKField,
#     out_covptot: IJKField,
#     out_fhpsl: IJKField,
#     out_fhpsn: IJKField,
#     out_fplsl: IJKField,
#     out_fplsn: IJKField,
#     out_tnd_q: IJKField,
#     out_tnd_qi: IJKField,
#     out_tnd_ql: IJKField,
#     out_tnd_t: IJKField,
#     tmp_aph_s: IJField,
#     dt: gtx.float,
# ):
#     _cloudsc2_next(
#         in_ap,
#         in_aph,
#         in_eta,
#         in_lu,
#         in_lude,
#         in_mfd,
#         in_mfu,
#         in_q,
#         in_qi,
#         in_ql,
#         in_qsat,
#         in_supsat,
#         in_t,
#         in_tnd_cml_q,
#         in_tnd_cml_qi,
#         in_tnd_cml_ql,
#         in_tnd_cml_t,
#         tmp_aph_s,
#         dt,
#         out=(
#             out_clc[:, :, :-1],
#             out_covptot[:, :, :-1],
#             out_fhpsl[:, :, :-1],
#             out_fhpsn[:, :, :-1],
#             out_fplsl[:, :, :-1],
#             out_fplsn[:, :, :-1],
#             out_tnd_q[:, :, :-1],
#             out_tnd_qi[:, :, :-1],
#             out_tnd_ql[:, :, :-1],
#             out_tnd_t[:, :, :-1],
#         ),
#     )
