# @Filename: tvdiff.py
# @Last modified by:   riener
# @Last modified time: 2019-03-01T13:27:43+01:00

#!/usr/bin/python
# AUTHOR:
#          Robert R. Lindner (UW-Madison)
#                            (www.robertrlindner.com)
#                            (robertrlindner@gmail.com)
#
#
# DATE:
#          October 30, 2013
#
#
# ISSUES:
#          These issues are still under active development:
#            (1) Code does not keep track of "dx" spacings
#            (2) For alpha approaching 0.0, the code does not
#                return the original naive finite-difference derivative.
#            (3) Native support for higher-order derivatives
#            (4) Shell (command line) operation
#
#
# DESCRIPTION:
#          Total-variation Tikhonov regularized numerical
#          differentiation of 1D data.
#
#          This Python code was adapted from the MATLAB routine of
#          Rick Chartrand (rickc@lanl.gov) presented in:
#          Chartrand 2011, "Numerical differentiation of noisy,nonsmooth data,"
#          ISRN Applied Mathematics, Vol. 2011, Article ID 164564 following the
#          algorithm from Chapter 8 in "Computational methods for
#          Inverse Problems" by Curtis R. Vogel.
#
#          Following the BSD license, appended below is the original
#          copywrite notice, list of conditions, BSD license text,
#          and disclaimer:
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# LEGAL:
#  Copyright notice:
# Copyright 2010. Los Alamos National Security, LLC. This material
# was produced under U.S. Government contract DE-AC52-06NA25396 for
# Los Alamos National Laboratory, which is operated by Los Alamos
# National Security, LLC, for the U.S. Department of Energy. The
# Government is granted for, itself and others acting on its
# behalf, a paid-up, nonexclusive, irrevocable worldwide license in
# this material to reproduce, prepare derivative works, and perform
# publicly and display publicly. Beginning five (5) years after
# (March 31, 2011) permission to assert copyright was obtained,
# subject to additional five-year worldwide renewals, the
# Government is granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute
# copies to the public, perform publicly and display publicly, and
# to permit others to do so. NEITHER THE UNITED STATES NOR THE
# UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
# SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
# EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
# RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
# ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
# REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
# RIGHTS.
#
# BSD License notice:
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#      Redistributions of source code must retain the above
#      copyright notice, this list of conditions and the following
#      disclaimer.
#      Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials
#      provided with the distribution.
#      Neither the name of Los Alamos National Security nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Code begins here:

# from matplotlib import pyplot as plt (used for debugging)
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator as lo


def TVdiff(data, dx=1.0, alph=0.1, beta=0.1, thresh=1E-4, max_iter=20):
    dxi = dx
    dx = 1.0

    # Data scaling
    scale_data = 1.0 / np.max(data)

    data = data * scale_data

    # Get length of data
    n = len(data)  # Number of data points
    data = np.matrix(np.reshape(data, [n, 1]))

    # Construct sparse differentiation matrix

    c = np.ones([n + 1]) / dx
    D = sparse.spdiags([-c, c], [0, 1], n, n + 1)
    D_initial = sparse.spdiags([-c, c], [0, 1], n - 1, n)
    DT = D.transpose()

    # Define anti-derivatives A and ATranspose
    def A(x):
        '''Antidifferentiation matrix'''
        out = np.cumsum(x).reshape([len(x), 1]) - 0.5 * (x + x[0, 0])
        return out[1:, :] * dx

    def AT(x):
        '''Transpose of antidifferentiation matrix'''
        out = np.sum(x) * np.ones([n + 1, 1])
        chunk1 = np.sum(x)
        chunk2 = np.cumsum(x).reshape([n, 1]) - x / 2.
        out[0, 0] = out[0, 0] - chunk1
        out[1:, :] = out[1:, :] - chunk2
        out = out * dx
        return out

    # Initial guess for u is naive derivative
    u = np.matrix(np.zeros([n+1, 1]))
    u[1:n] = D_initial * data

    # First data item
    ofst = data[0]
    ATb = AT(ofst - data)

    # Main loop
    i = 0
    norm = 1.0
    while (norm > thresh) and (i < max_iter):
        diags = 1. / np.sqrt(np.array(D * u)**2 + beta**2)

        Q = sparse.spdiags(diags.ravel(), 0, n, n)

        # Hessian term
        L = dx * DT * Q * D

        # Gradient of functional.
        g = AT(A(u)) + ATb + alph * L * u

        def f(x):
            x = x.reshape([len(x), 1])
            a = alph * L * x + AT(A(x))
            return a

        operator = lo((n+1, n+1), f, dtype='float')
        s = cg(operator, g)[0]

        # DEBUGGING:
        # print plt.plot(np.array((operator * s)).ravel(), np.array((g)).ravel())
        # plt.show()
        # quit()
        #######################

        s = np.reshape(s, [len(s), 1])
        u = u - s

        # Test the convergence condition
        s_norm = np.sqrt(np.sum(np.array(s).ravel()**2))
        u_norm = np.sqrt(np.sum(np.array(u).ravel()**2))
        norm = s_norm / u_norm

        i = i + 1
    u = u[1:]  # Clip off the first element

    return np.array(u).ravel() / scale_data / dxi


if __name__ == "__main__":
    print('Not running from command line yet!...')
