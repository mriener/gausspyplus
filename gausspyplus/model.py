from dataclasses import dataclass
from typing import List

import numpy as np

from gausspyplus.spectrum import Spectrum
from gausspyplus.utils.fit_quality_checks import goodness_of_fit, check_residual_for_normality
from gausspyplus.utils.gaussian_functions import number_of_gaussian_components, split_params, combined_gaussian


@dataclass
class Model:
    spectrum: Spectrum
    _parameters: List = None
    _n_components: int = 0  # ncomps_fit
    _amps: List = None
    _fwhms: List = None
    _means: List = None
    _parameter_uncertainties: List = None  # "params_errs"
    _modelled_intensity_values: np.ndarray = None  # best_fit_final
    _residual: np.ndarray = None  # residual
    _rchi2: float = None  # rchi2
    _aicc: float = None  # aicc
    new_best_fit: bool = False  # new_fit
    parameters_min_values: List = None  # params_min
    parameters_max_values: List = None  # params_max
    _pvalue: float = None  # pvalue
    _quality_control: List = None  # quality_control

    @property
    def parameters(self) -> List:
        return self._parameters

    @parameters.setter
    def parameters(self, values: List) -> None:
        self._parameters = values
        if len(self._parameters) % 3 != 0:
            raise Exception("One or more fit parameters are missing")
        ncomps = number_of_gaussian_components(params=values)
        self._amps, self._fwhms, self._means = split_params(params=values, ncomps=ncomps)
        self._n_components = ncomps
        self._modelled_intensity_values = combined_gaussian(
            amps=self._amps,
            fwhms=self._fwhms,
            means=self._means,
            x=self.spectrum.channels
        )
        self._residual = self.spectrum.intensity_values - self._modelled_intensity_values
        self._rchi2, self._aicc = goodness_of_fit(
            data=self.spectrum.intensity_values,
            best_fit_final=self._modelled_intensity_values,
            errors=self.spectrum.noise_values,
            ncomps_fit=self.n_components,
            mask=self.spectrum.signal_mask,
            get_aicc=True
        )
        self._pvalue = check_residual_for_normality(
            data=self._residual,
            errors=self.spectrum.noise_values,
            mask=self.spectrum.signal_mask,
            noise_spike_mask=self.spectrum.noise_spike_mask
        )

    @property
    def parameter_uncertainties(self) -> List:
        return [] if self._parameter_uncertainties is None else self._parameter_uncertainties

    @parameter_uncertainties.setter
    def parameter_uncertainties(self, values: List) -> None:
        self._parameter_uncertainties = values

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def amps(self) -> List:
        return self._amps

    @property
    def fwhms(self) -> List:
        return self._fwhms

    @property
    def means(self) -> List:
        return self._means

    @property
    def modelled_intensity_values(self) -> np.ndarray:
        return self._modelled_intensity_values

    @property
    def residual(self) -> np.ndarray:
        return self._residual

    @property
    def rchi2(self) -> float:
        return self._rchi2

    @property
    def aicc(self) -> float:
        return self._aicc

    @property
    def pvalue(self):
        return self._pvalue

    @property
    def quality_control(self) -> List:
        self._quality_control = [] if self._quality_control is None else self._quality_control
        return self._quality_control

    @quality_control.setter
    def quality_control(self, value: List):
        self._quality_control = value
