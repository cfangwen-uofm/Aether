// Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
// Full license can be found in License.md

#ifndef INCLUDE_CUBESPHERE_H_
#define INCLUDE_CUBESPHERE_H_

#include "aether.h"
#include <memory>
#include <armadillo>

/*************************************************
 * \brief A namespace with all quad sphere grid logic.
 *************************************************/
namespace CubeSphere {

/// The normalized origins of each face of the cube (i.e. corner)
static const arma_mat ORIGINS = {
    {-1.0, -1.0, -1.0},
    { 1.0, -1.0, -1.0},
    { 1.0,  1.0, -1.0},
    {-1.0,  1.0, -1.0},
    {-1.0, -1.0, -1.0},
    { 1.0, -1.0,  1.0},
};

/// Normalized right steps in cube
static const arma_mat RIGHTS = {
		{ 2.0,  0.0, 0.0},
		{ 0.0,  2.0, 0.0},
		{-2.0,  0.0, 0.0},
		{ 0.0, -2.0, 0.0},
		{ 0.0,  2.0, 0.0},
		{ 0.0,  2.0, 0.0}
};

/// Normalized right steps in cube
static const arma_mat UPS = {
		{ 0.0, 0.0, 2.0},
		{ 0.0, 0.0, 2.0},
		{ 0.0, 0.0, 2.0},
		{ 0.0, 0.0, 2.0},
		{ 2.0, 0.0, 0.0},
		{-2.0, 0.0, 0.0}
};

} // CubeSphere::

namespace Cubesphere_tools
{	
	struct projection_struct
	{
		arma_mat gradLR;
		arma_mat gradDU;
		arma_mat R;
		arma_mat L;
		arma_mat U;
		arma_mat D;
	};
	arma_vec limiter_mc(arma_vec &left,arma_vec &right,int64_t nPts,int64_t nGCs);
	void print(arma_vec values);
	arma_vec calc_grad_1d(arma_vec &values,arma_vec &x,int64_t nPts,int64_t nGCs);
	arma_mat calc_grad(arma_mat values,arma_mat x,int64_t nGCs,bool DoX);
	arma_mat project_from_left(arma_mat values,arma_mat gradients,arma_mat x_centers,arma_mat x_edges,int64_t nGCs);
	arma_mat project_from_right(arma_mat values,arma_mat gradients,arma_mat x_centers,arma_mat x_edges,int64_t nGCs);
	arma_vec limiter_value(arma_vec projected,arma_vec values,int64_t nPts,int64_t nGCs);
	projection_struct project_to_edges(arma_mat &values,arma_mat &x_centers, arma_mat &x_edges,arma_mat &y_centers, arma_mat &y_edges,int64_t nGCs);
}

#endif  // INCLUDE_CUBESPHERE_H_