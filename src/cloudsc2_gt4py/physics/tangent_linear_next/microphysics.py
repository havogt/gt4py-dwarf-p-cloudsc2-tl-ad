# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from itertools import repeat
from functools import cached_property
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage, zeros
from ifs_physics_common.utils.numpyx import assign

import jax

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from datetime import timedelta
    from typing import Dict, Optional, Union

    from gt4py.cartesian import StencilObject

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import (
        NDArrayLike,
        NDArrayLikeDict,
        ParameterDict,
        PropertyDict,
    )

from gt4py.next.embedded import (
    operators as embedded_operators,
    context as embedded_context,
)
from gt4py.next import common
from ..common import next as common_next
from ..nonlinear_next._stencils import cloudsc2_next

I2 = common_next.I2
J2 = common_next.J2
K2 = common_next.K2


def tangent_linear_operator(fun): ...


class Cloudsc2TLnext(ImplicitTendencyComponent):
    klevel: NDArrayLike

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        lphylin: bool,
        ldrain1d: bool,
        yoethf_parameters: Optional[ParameterDict] = None,
        yomcst_parameters: Optional[ParameterDict] = None,
        yrecld_parameters: Optional[ParameterDict] = None,
        yrecldp_parameters: Optional[ParameterDict] = None,
        yrephli_parameters: Optional[ParameterDict] = None,
        yrncl_parameters: Optional[ParameterDict] = None,
        yrphnc_parameters: Optional[ParameterDict] = None,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(
            computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config
        )

        nk = self.computational_grid.grids[I, J, K].shape[2]
        self.klevel = zeros(
            self.computational_grid, (K,), gt4py_config=self.gt4py_config, dtype="int"
        )
        assign(self.klevel[:], np.arange(0, nk + 1))

        externals: Dict[str, Union[bool, float, int]] = {}
        externals.update(yoethf_parameters or {})
        externals.update(yomcst_parameters or {})
        externals.update(yrecld_parameters or {})
        externals.update(yrecldp_parameters or {})
        externals.update(yrephli_parameters or {})
        externals.update(yrncl_parameters or {})
        externals.update(yrphnc_parameters or {})
        externals.update(
            {
                "ICALL": 0,
                "LPHYLIN": lphylin,
                "LDRAIN1D": ldrain1d,
                "NLEV": nk,
                "ZEPS1": 1e-12,
                "ZEPS2": 1e-10,
                "ZQMAX": 0.5,
                "ZSCAL": 0.9,
            }
        )

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "f_eta": {"grid": (K,), "units": ""},
            "f_aph": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_aph_i": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_ap": {"grid": (I, J, K), "units": "Pa"},
            "f_ap_i": {"grid": (I, J, K), "units": "Pa"},
            "f_q": {"grid": (I, J, K), "units": "g g^-1"},
            "f_q_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_t": {"grid": (I, J, K), "units": "K"},
            "f_t_i": {"grid": (I, J, K), "units": "K"},
            "f_ql": {"grid": (I, J, K), "units": "g g^-1"},
            "f_ql_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lude": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lude_i": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lu": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lu_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_mfu": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfu_i": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd_i": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_tnd_cml_t": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_t_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_supsat": {"grid": (I, J, K), "units": "g g^-1"},
            "f_supsat_i": {"grid": (I, J, K), "units": "g g^-1"},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
            "f_t": {"grid": (I, J, K), "units": "K s^-1"},
            "f_t_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_q": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_q_i": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_ql": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_ql_i": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_qi": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_qi_i": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "f_clc": {"grid": (I, J, K), "units": ""},
            "f_clc_i": {"grid": (I, J, K), "units": ""},
            "f_fhpsl": {"grid": (I, J, K - 1 / 2), "units": "J m^-2 s^-1"},
            "f_fhpsl_i": {"grid": (I, J, K - 1 / 2), "units": "J m^-2 s^-1"},
            "f_fhpsn": {"grid": (I, J, K - 1 / 2), "units": "J m^-2 s^-1"},
            "f_fhpsn_i": {"grid": (I, J, K - 1 / 2), "units": "J m^-2 s^-1"},
            "f_fplsl": {"grid": (I, J, K - 1 / 2), "units": "Kg m^-2 s^-1"},
            "f_fplsl_i": {"grid": (I, J, K - 1 / 2), "units": "Kg m^-2 s^-1"},
            "f_fplsn": {"grid": (I, J, K - 1 / 2), "units": "Kg m^-2 s^-1"},
            "f_fplsn_i": {"grid": (I, J, K - 1 / 2), "units": "Kg m^-2 s^-1"},
            "f_covptot": {"grid": (I, J, K), "units": ""},
            "f_covptot_i": {"grid": (I, J, K), "units": ""},
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: dict[str, bool],
    ) -> None:
        use_jax = True

        primals = (
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_ap"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_aph"]),
            # common_next.as_field(K2, use_jax=use_jax)(state["f_eta"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_lu"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_lude"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_mfd"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_mfu"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_q"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_qi"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_ql"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_qsat"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_supsat"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_t"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_q"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_qi"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_ql"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_t"]),
            common_next.as_field(I2, J2, use_jax=use_jax)(state["f_aph"][..., -1]),
        )
        tangents = (
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_ap_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_aph_i"]),
            # common_next.as_field(K2, use_jax=use_jax)(state["f_eta_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_lu_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_lude_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_mfd_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_mfu_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_q_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_qi_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_ql_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_qsat_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_supsat_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_t_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_q_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_qi_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_ql_i"]),
            common_next.as_field(I2, J2, K2, use_jax=use_jax)(state["f_tnd_cml_t_i"]),
            common_next.as_field(I2, J2, use_jax=use_jax)(state["f_aph_i"][..., -1]),
        )

        shape = out_diagnostics["f_clc"].shape
        offset_provider = {"K": K2}
        domain = common.domain({I2: shape[0], J2: shape[1], K2: shape[2] - 1})
        with embedded_context.new_context(
            closure_column_range=embedded_operators._get_vertical_range(domain),
            offset_provider=offset_provider,
        ) as ctx:

            def cloudsc2_wrapper(
                in_ap,
                in_aph,
                in_lu,
                in_lude,
                in_mfd,
                in_mfu,
                in_q,
                in_qi,
                in_ql,
                in_qsat,
                in_supsat,
                in_t,
                in_tnd_cml_q,
                in_tnd_cml_qi,
                in_tnd_cml_ql,
                in_tnd_cml_t,
                tmp_aph_s,
            ):
                return cloudsc2_next._cloudsc2_next(
                    in_ap=in_ap,
                    in_aph=in_aph,
                    in_eta=common_next.as_field(K2, use_jax=use_jax)(state["f_eta"]),
                    in_lu=in_lu,
                    in_lude=in_lude,
                    in_mfd=in_mfd,
                    in_mfu=in_mfu,
                    in_q=in_q,
                    in_qi=in_qi,
                    in_ql=in_ql,
                    in_qsat=in_qsat,
                    in_supsat=in_supsat,
                    in_t=in_t,
                    in_tnd_cml_q=in_tnd_cml_q,
                    in_tnd_cml_qi=in_tnd_cml_qi,
                    in_tnd_cml_ql=in_tnd_cml_ql,
                    in_tnd_cml_t=in_tnd_cml_t,
                    tmp_aph_s=tmp_aph_s,
                    dt=self.gt4py_config.dtypes.float(timestep.total_seconds()),
                )

            def execute(primals, tangents):
                return jax.jvp(cloudsc2_wrapper, primals, tangents)
                # y, vjp_fun = jax.vjp(cloudsc2_wrapper, *primals)
                # return y, vjp_fun(tangents)

            primals_out, tangents_out = ctx.run(execute, primals, tangents)

            (
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_clc"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_covptot"])[
                    :, :, :-1
                ],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fhpsl"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fhpsn"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fplsl"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fplsn"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_q"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_qi"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_ql"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_t"])[:, :, :-1],
            ) = primals_out
            (
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_clc_i"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_covptot_i"])[
                    :, :, :-1
                ],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fhpsl_i"])[
                    :, :, :-1
                ],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fhpsn_i"])[
                    :, :, :-1
                ],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fplsl_i"])[
                    :, :, :-1
                ],
                common_next.as_field(I2, J2, K2)(out_diagnostics["f_fplsn_i"])[
                    :, :, :-1
                ],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_q_i"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_qi_i"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_ql_i"])[:, :, :-1],
                common_next.as_field(I2, J2, K2)(out_tendencies["f_t_i"])[:, :, :-1],
            ) = tangents_out

        # self.cloudsc2(
        #     in_ap=state["f_ap"],
        #     in_ap_i=state["f_ap_i"],
        #     in_aph=state["f_aph"],
        #     in_aph_i=state["f_aph_i"],
        #     in_eta=state["f_eta"],
        #     in_lu=state["f_lu"],
        #     in_lu_i=state["f_lu_i"],
        #     in_lude=state["f_lude"],
        #     in_lude_i=state["f_lude_i"],
        #     in_mfd=state["f_mfd"],
        #     in_mfd_i=state["f_mfd_i"],
        #     in_mfu=state["f_mfu"],
        #     in_mfu_i=state["f_mfu_i"],
        #     in_q=state["f_q"],
        #     in_q_i=state["f_q_i"],
        #     in_qi=state["f_qi"],
        #     in_qi_i=state["f_qi_i"],
        #     in_ql=state["f_ql"],
        #     in_ql_i=state["f_ql_i"],
        #     in_qsat=state["f_qsat"],
        #     in_qsat_i=state["f_qsat_i"],
        #     in_supsat=state["f_supsat"],
        #     in_supsat_i=state["f_supsat_i"],
        #     in_t=state["f_t"],
        #     in_t_i=state["f_t_i"],
        #     in_tnd_cml_q=state["f_tnd_cml_q"],
        #     in_tnd_cml_q_i=state["f_tnd_cml_q_i"],
        #     in_tnd_cml_qi=state["f_tnd_cml_qi"],
        #     in_tnd_cml_qi_i=state["f_tnd_cml_qi_i"],
        #     in_tnd_cml_ql=state["f_tnd_cml_ql"],
        #     in_tnd_cml_ql_i=state["f_tnd_cml_ql_i"],
        #     in_tnd_cml_t=state["f_tnd_cml_t"],
        #     in_tnd_cml_t_i=state["f_tnd_cml_t_i"],
        #     tmp_aph_s = state["f_aph"][..., -1]
        #     tmp_aph_s_i = state["f_aph_i"][..., -1]
        #     out_clc=out_diagnostics["f_clc"],
        #     out_clc_i=out_diagnostics["f_clc_i"],
        #     out_covptot=out_diagnostics["f_covptot"],
        #     out_covptot_i=out_diagnostics["f_covptot_i"],
        #     out_fhpsl=out_diagnostics["f_fhpsl"],
        #     out_fhpsl_i=out_diagnostics["f_fhpsl_i"],
        #     out_fhpsn=out_diagnostics["f_fhpsn"],
        #     out_fhpsn_i=out_diagnostics["f_fhpsn_i"],
        #     out_fplsl=out_diagnostics["f_fplsl"],
        #     out_fplsl_i=out_diagnostics["f_fplsl_i"],
        #     out_fplsn=out_diagnostics["f_fplsn"],
        #     out_fplsn_i=out_diagnostics["f_fplsn_i"],
        #     out_tnd_q=out_tendencies["f_q"],
        #     out_tnd_q_i=out_tendencies["f_q_i"],
        #     out_tnd_qi=out_tendencies["f_qi"],
        #     out_tnd_qi_i=out_tendencies["f_qi_i"],
        #     out_tnd_ql=out_tendencies["f_ql"],
        #     out_tnd_ql_i=out_tendencies["f_ql_i"],
        #     out_tnd_t=out_tendencies["f_t"],
        #     out_tnd_t_i=out_tendencies["f_t_i"],
        #     dt=self.gt4py_config.dtypes.float(timestep.total_seconds()),
        #     origin=(0, 0, 0),
        #     domain=self.computational_grid.grids[I, J, K - 1 / 2].shape,
        #     validate_args=self.gt4py_config.validate_args,
        #     exec_info=self.gt4py_config.exec_info,
        # )
