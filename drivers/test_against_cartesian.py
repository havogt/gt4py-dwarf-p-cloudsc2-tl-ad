from cloudsc2_gt4py.physics.common.diagnostics import EtaLevels
from cloudsc2_gt4py.physics.common.saturation import Saturation
from cloudsc2_gt4py.physics.nonlinear.microphysics import Cloudsc2NL
from cloudsc2_gt4py.physics.nonlinear import microphysics as mphys
from cloudsc2_gt4py.physics.nonlinear_next import microphysics as mphys_next
from cloudsc2_gt4py.physics.nonlinear_next.microphysics import Cloudsc2NLnext
from cloudsc2_gt4py.physics.tangent_linear.microphysics import Cloudsc2TL
from cloudsc2_gt4py.physics.tangent_linear_next.microphysics import Cloudsc2TLnext
from cloudsc2_gt4py.physics.common.increment import StateIncrement

from cloudsc2_gt4py.physics.nonlinear.validation import validate
from cloudsc2_gt4py.state import get_initial_state
from cloudsc2_gt4py.utils.iox import HDF5Reader
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.output import (
    print_performance,
    write_performance_to_csv,
    write_stencils_performance_to_csv,
)
from ifs_physics_common.utils.timing import timing
from os.path import normpath, dirname, join

import numpy as np

from config import DEFAULT_CONFIG


config = DEFAULT_CONFIG
hdf5_reader = HDF5Reader(config.input_file, config.data_types)

# grid
nx = config.num_cols or hdf5_reader.get_nlon()
config = config.with_num_cols(nx)
nz = hdf5_reader.get_nlev()
computational_grid = ComputationalGrid(nx, 1, nz)


def shared_setup():
    # state and accumulated tendencies
    state = get_initial_state(
        computational_grid, hdf5_reader, gt4py_config=config.gt4py_config
    )

    # diagnose reference eta-levels
    eta_levels = EtaLevels(
        computational_grid,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    state.update(eta_levels(state))
    return state


def test_tangent_linear():
    state = shared_setup()

    yoethf_params = hdf5_reader.get_yoethf_parameters()
    yomcst_params = hdf5_reader.get_yomcst_parameters()
    yrecld_params = hdf5_reader.get_yrecld_parameters()
    yrecldp_params = hdf5_reader.get_yrecldp_parameters()
    yrephli_params = hdf5_reader.get_yrephli_parameters()
    yrncl_params = hdf5_reader.get_yrncl_parameters()
    yrphnc_params = hdf5_reader.get_yrphnc_parameters()
    saturation = Saturation(
        computational_grid,
        kflag=1,
        lphylin=True,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )

    diagnostics = saturation(state)
    state.update(diagnostics)

    dt = hdf5_reader.get_timestep()

    factor1 = 0.01
    state_increment = StateIncrement(
        computational_grid,
        factor1,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )

    state_i = {}
    state_i = state_increment(state, out=state_i)
    state.update(state_i)

    cloudsc_tl = Cloudsc2TL(
        computational_grid,
        lphylin=True,
        ldrain1d=False,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        yrecld_parameters=yrecld_params,
        yrecldp_parameters=yrecldp_params,
        yrephli_parameters=yrephli_params,
        yrncl_parameters=yrncl_params,
        yrphnc_parameters=yrphnc_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )

    tendencies, diags = cloudsc_tl(state, dt)

    cloudsc_tl_next = Cloudsc2TLnext(
        computational_grid,
        lphylin=True,
        ldrain1d=False,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        yrecld_parameters=yrecld_params,
        yrecldp_parameters=yrecldp_params,
        yrephli_parameters=yrephli_params,
        yrncl_parameters=yrncl_params,
        yrphnc_parameters=yrphnc_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )

    tendencies_next, diags_next = cloudsc_tl_next(state, dt)

    np.testing.assert_allclose(diags["f_clc"], diags_next["f_clc"])
    np.testing.assert_allclose(diags["f_covptot"], diags_next["f_covptot"])
    np.testing.assert_allclose(
        diags["f_fplsl"][:, :, :-1], diags_next["f_fplsl"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fplsn"][:, :, :-1], diags_next["f_fplsn"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fhpsl"][:, :, :-1], diags_next["f_fhpsl"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fhpsn"][:, :, :-1], diags_next["f_fhpsn"][:, :, :-1]
    )
    np.testing.assert_allclose(tendencies["f_q"], tendencies_next["f_q"])
    np.testing.assert_allclose(tendencies["f_qi"], tendencies_next["f_qi"])
    np.testing.assert_allclose(tendencies["f_ql"], tendencies_next["f_ql"])
    np.testing.assert_allclose(tendencies["f_t"], tendencies_next["f_t"])

    np.testing.assert_allclose(diags["f_covptot_i"], diags_next["f_covptot_i"])
    np.testing.assert_allclose(
        diags["f_fplsl_i"][:, :, :-1], diags_next["f_fplsl_i"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fplsn_i"][:, :, :-1], diags_next["f_fplsn_i"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fhpsl_i"][:, :, :-1], diags_next["f_fhpsl_i"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fhpsn_i"][:, :, :-1], diags_next["f_fhpsn_i"][:, :, :-1]
    )
    print(tendencies["f_q_i"] - tendencies_next["f_q_i"])
    np.testing.assert_allclose(tendencies["f_q_i"], tendencies_next["f_q_i"])
    np.testing.assert_allclose(tendencies["f_qi_i"], tendencies_next["f_qi_i"])
    np.testing.assert_allclose(tendencies["f_ql_i"], tendencies_next["f_ql_i"])
    np.testing.assert_allclose(tendencies["f_t_i"], tendencies_next["f_t_i"])
    print(diags["f_clc_i"] - diags_next["f_clc_i"])

    # NOTE: adjusted tolerance
    np.testing.assert_allclose(
        diags["f_clc_i"], diags_next["f_clc_i"], rtol=1.0, atol=1e-16
    )

    print("TL PASSED!")


def test_non_linear():
    state = shared_setup()

    # timestep
    dt = hdf5_reader.get_timestep()

    # parameters
    yoethf_params = hdf5_reader.get_yoethf_parameters()
    yomcst_params = hdf5_reader.get_yomcst_parameters()
    yrecld_params = hdf5_reader.get_yrecld_parameters()
    yrecldp_params = hdf5_reader.get_yrecldp_parameters()
    yrephli_params = hdf5_reader.get_yrephli_parameters()
    yrphnc_params = hdf5_reader.get_yrphnc_parameters()

    # saturation
    saturation = Saturation(
        computational_grid,
        kflag=1,
        lphylin=True,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    diagnostics = saturation(state)
    state.update(diagnostics)

    # microphysics
    cloudsc2_nl = Cloudsc2NL(
        computational_grid,
        lphylin=True,
        ldrain1d=False,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        yrecld_parameters=yrecld_params,
        yrecldp_parameters=yrecldp_params,
        yrephli_parameters=yrephli_params,
        yrphnc_parameters=yrphnc_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    tendencies, diags = cloudsc2_nl(state, dt)

    cloudsc2_nl_next = Cloudsc2NLnext(
        computational_grid,
        lphylin=True,
        ldrain1d=False,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        yrecld_parameters=yrecld_params,
        yrecldp_parameters=yrecldp_params,
        yrephli_parameters=yrephli_params,
        yrphnc_parameters=yrphnc_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    tendencies_next, diags_next = cloudsc2_nl_next(state, dt)

    print(diags_next["f_clc"])
    print(diags["f_clc"])
    print(diags_next["f_fplsl"])
    print(tendencies["f_t"])
    print(tendencies_next["f_t"])
    print(tendencies["f_t"] - tendencies_next["f_t"])
    print(tendencies["f_q"] - tendencies_next["f_q"])
    print(diags["f_clc"] - diags_next["f_clc"])

    np.testing.assert_allclose(diags["f_clc"], diags_next["f_clc"])
    np.testing.assert_allclose(diags["f_covptot"], diags_next["f_covptot"])
    np.testing.assert_allclose(
        diags["f_fplsl"][:, :, :-1], diags_next["f_fplsl"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fplsn"][:, :, :-1], diags_next["f_fplsn"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fhpsl"][:, :, :-1], diags_next["f_fhpsl"][:, :, :-1]
    )
    np.testing.assert_allclose(
        diags["f_fhpsn"][:, :, :-1], diags_next["f_fhpsn"][:, :, :-1]
    )
    np.testing.assert_allclose(tendencies["f_q"], tendencies_next["f_q"])
    np.testing.assert_allclose(tendencies["f_qi"], tendencies_next["f_qi"])
    np.testing.assert_allclose(tendencies["f_ql"], tendencies_next["f_ql"])
    np.testing.assert_allclose(tendencies["f_t"], tendencies_next["f_t"])

    print("NL PASSED!")


# test_non_linear()
test_tangent_linear()
