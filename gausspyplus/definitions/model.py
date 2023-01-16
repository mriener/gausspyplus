from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from gausspyplus.definitions.spectrum import Spectrum
from gausspyplus.decomposition.fit_quality_checks import (
    goodness_of_fit,
    check_residual_for_normality,
    get_indices_of_blended_components,
    get_number_of_blended_components,
    negative_residuals,
)
from gausspyplus.decomposition.gaussian_functions import (
    number_of_gaussian_components,
    split_params,
    multi_component_gaussian_model,
)


@dataclass
class Model:
    spectrum: Spectrum
    _parameters: List = None
    _n_components: int = 0  # ncomps_fit
    _amps: List = None
    _fwhms: List = None
    _means: List = None
    _amps_uncertainties: List = None
    _fwhms_uncertainties: List = None
    _means_uncertainties: List = None
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
    _log_of_successful_refits: List = None

    @property
    def parameters(self) -> List:
        return self._parameters

    @parameters.setter
    def parameters(self, values: List) -> None:
        self._parameters = values
        if len(self._parameters) % 3 != 0:
            raise Exception("One or more fit parameters are missing")
        ncomps = number_of_gaussian_components(params=values)
        # TODO: Replace split_params with np.split?
        self._amps, self._fwhms, self._means = split_params(params=values, ncomps=ncomps)
        self._n_components = ncomps
        self._modelled_intensity_values = multi_component_gaussian_model(
            amps=self._amps,
            fwhms=self._fwhms,
            means=self._means,
            x=self.spectrum.channels,
        )
        self._residual = self.spectrum.intensity_values - self._modelled_intensity_values
        self._rchi2, self._aicc = goodness_of_fit(
            data=self.spectrum.intensity_values,
            best_fit_final=self._modelled_intensity_values,
            errors=self.spectrum.noise_values,
            ncomps_fit=self.n_components,
            mask=self.spectrum.signal_mask,
            get_aicc=True,
        )
        self._pvalue = check_residual_for_normality(
            data=self._residual,
            errors=self.spectrum.noise_values,
            mask=self.spectrum.signal_mask,
            noise_spike_mask=self.spectrum.noise_spike_mask,
        )

    @property
    def parameter_uncertainties(self) -> List:
        return [] if self._parameter_uncertainties is None else self._parameter_uncertainties

    @parameter_uncertainties.setter
    def parameter_uncertainties(self, values: List) -> None:
        if len(values) % 3 != 0:
            raise Exception("One or more fit uncertainty parameters are missing")
        self._parameter_uncertainties = values
        ncomps = number_of_gaussian_components(params=values)
        (
            self._amps_uncertainties,
            self._fwhms_uncertainties,
            self._means_uncertainties,
        ) = split_params(params=values, ncomps=ncomps)

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
    def amps_uncertainties(self) -> List:
        return self._amps_uncertainties

    @property
    def fwhms_uncertainties(self) -> List:
        return self._fwhms_uncertainties

    @property
    def means_uncertainties(self) -> List:
        return self._means_uncertainties

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

    def log_in_case_of_successful_refit(
        self,
        mode: Literal["positive_residual_peak", "negative_residual_peak", "broad", "blended"],
    ) -> None:
        if self.new_best_fit:
            log_of_successful_refits = self.log_of_successful_refits
            log_of_successful_refits.append(
                {
                    "positive_residual_peak": 1,
                    "negative_residual_peak": 2,
                    "broad": 3,
                    "blended": 4,
                }[mode]
            )
            self._log_of_successful_refits = log_of_successful_refits

    @property
    def log_of_successful_refits(self) -> List:
        return [] if self._log_of_successful_refits is None else self._log_of_successful_refits

    @log_of_successful_refits.setter
    def log_of_successful_refits(self, log: List) -> None:
        self._log_of_successful_refits = log

    def indices_of_blended_components(self, separation_factor: float = 0.8493218002991817):
        return get_indices_of_blended_components(self.parameters, separation_factor)

    def number_of_blended_components(self, separation_factor: float = 0.8493218002991817):
        return get_number_of_blended_components(self.parameters, separation_factor)

    def number_of_negative_residual_peaks(self, neg_res_snr: float = 3.0):
        return negative_residuals(self.spectrum.intensity_values, self.residual, self.spectrum.rms_noise, neg_res_snr)
