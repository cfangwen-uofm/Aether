// Copyright 2023, the Aether Development Team (see doc/dev_team.md for members)
// Full license can be found in License.md

// Initial version: F. Cheng, July 2023

#include "../include/aether.h"

struct projection_struct
{

  arma_mat gradLR;
  arma_mat gradDU;
  arma_mat R;
  arma_mat L;
  arma_mat U;
  arma_mat D;
};

// ---------------------------------------------------------
//
// ---------------------------------------------------------

arma_vec limiter_mc(arma_vec &left,
                    arma_vec &right,
                    int64_t nPts,
                    int64_t nGCs)
{

  precision_t beta = 0.8;

  arma_vec s = left % right;
  arma_vec combined = (left + right) * 0.5;

  left = left * beta;
  right = right * beta;
  arma_vec limited = left;

  for (int64_t i = 1; i < nPts + 2 * nGCs - 1; i++)
  {
    if (s(i) < 0)
    {
      // Sign < 0 means opposite signed left and right:
      limited(i) = 0.0;
    }
    else
    {
      if (left(i) > 0 && right(i) > 0)
      {
        if (right(i) < limited(i))
          limited(i) = right(i);
        if (combined(i) < limited(i))
          limited(i) = combined(i);
      }
      else
      {
        if (right(i) > limited(i))
          limited(i) = right(i);
        if (combined(i) > limited(i))
          limited(i) = combined(i);
      }
    }
  }
  return limited;
}

void print(arma_vec values)
{
  int64_t nP = values.n_elem;
  for (int64_t i = 0; i < nP; i++)
    std::cout << values(i) << " ";
  std::cout << "\n";
}

// ---------------------------------------------------------
// calc gradients at centers
//   - values and x defined at centers
// ---------------------------------------------------------

arma_vec calc_grad_1d(arma_vec &values,
                      arma_vec &x,
                      int64_t nPts,
                      int64_t nGCs)
{

  arma_vec gradients = values * 0.0;
  arma_vec gradL = values * 0.0;
  arma_vec gradR = values * 0.0;

  precision_t factor1 = 0.625;
  precision_t factor2 = 0.0416667;
  precision_t h;

  int64_t i;
  arma_vec hv = values * 0.0;

  i = nGCs - 1;
  h = 2.0 / (x(i + 1) - x(i));
  gradR(i) = h * (factor1 * (values(i + 1) - values(i)) -
                  factor2 * (values(i + 2) - values(i - 1)));
  gradL(i) = (values(i) - values(i - 1)) / (x(i) - x(i - 1));

  // This is attempting to vectorize the problem, but it seems to be slower?
  //  int64_t iS = nGCs;
  //  int64_t iE = nPts + nGCs - 1;
  //  hv.rows(iS, iE) = 2.0 / (x.rows(iS, iE) - x.rows(iS-1, iE-1));
  //  gradL.rows(iS, iE) = hv.rows(iS,iE) % (factor1 * (values.rows(iS, iE) -
  //						    values.rows(iS-1, iE-1)) -
  //					 factor2 * (values.rows(iS+1, iE+1) -
  //						    values.rows(iS-2, iE-2)));
  //  hv.rows(iS, iE) = 2.0 / (x.rows(iS+1, iE+1) - x.rows(iS, iE));
  //  gradR.rows(iS, iE) = hv.rows(iS,iE) % (factor1 * (values.rows(iS+1, iE+1) -
  //						    values.rows(iS, iE)) -
  //					 factor2 * (values.rows(iS+2, iE+2) -
  //						    values.rows(iS-1, iE-1)));

  for (i = nGCs; i < nPts + nGCs; i++)
  {
    h = 2.0 / (x(i) - x(i - 1));
    gradL(i) = h * (factor1 * (values(i) - values(i - 1)) -
                    factor2 * (values(i + 1) - values(i - 2)));
    h = 2.0 / (x(i + 1) - x(i));
    gradR(i) = h * (factor1 * (values(i + 1) - values(i)) -
                    factor2 * (values(i + 2) - values(i - 1)));
  }
  i = nPts + nGCs;
  h = 2.0 / (x(i) - x(i - 1));
  gradL(i) = h * (factor1 * (values(i) - values(i - 1)) -
                  factor2 * (values(i + 1) - values(i - 2)));
  gradR(i) = (values(i + 1) - values(i)) / (x(i + 1) - x(i));

  gradients = limiter_mc(gradL, gradR, nPts, nGCs);

  return gradients;
}

// ---------------------------------------------------------
// calc gradients at centers for 2d matrices
//   - values and x defined at centers
// ---------------------------------------------------------

arma_mat calc_grad(arma_mat values,
                   arma_mat x,
                   int64_t nGCs,
                   bool DoX)
{

  arma_mat v2d, x2d;

  if (DoX)
  {
    v2d = values;
    x2d = x;
  }
  else
  {
    v2d = values.t();
    x2d = x.t();
  }

  int64_t nX = v2d.n_rows;
  int64_t nY = v2d.n_cols;
  arma_mat grad2d = v2d * 0.0;

  int64_t nPts = nX - 2 * nGCs;
  arma_vec values1d(nX);
  arma_vec x1d(nX);
  for (int64_t j = 1; j < nY - 1; j++)
  {
    values1d = v2d.col(j);
    x1d = x2d.col(j);
    grad2d.col(j) = calc_grad_1d(values1d, x1d, nPts, nGCs);
  }

  arma_mat gradients;

  if (DoX)
  {
    gradients = grad2d;
  }
  else
  {
    gradients = grad2d.t();
  }
  return gradients;
}

// ---------------------------------------------------------
// Project gradients + values to the right face, from the left
//   returned values are on the i - 1/2 edges
//     (between i-1 and i cell center)
// ---------------------------------------------------------

arma_mat project_from_left(arma_mat values,
                           arma_mat gradients,
                           arma_mat x_centers,
                           arma_mat x_edges,
                           int64_t nGCs)
{

  int64_t nX = values.n_rows;
  int64_t nY = values.n_cols;

  // Define at edges:
  arma_mat projected(nX + 1, nY);
  projected.zeros();

  // no gradient in the 0 or iEnd cells
  for (int64_t j = 0; j < nY; j++)
  {
    for (int64_t i = 1; i < nX - 1; i++)
    {
      projected(i + 1, j) = values(i, j) +
                            gradients(i, j) * (x_edges(i + 1, j) - x_centers(i, j));
    }
    projected(1, j) = projected(2, j);
    projected(0, j) = projected(1, j);
    projected(nX, j) = projected(nX - 1, j);
  }
  return projected;
}

// ---------------------------------------------------------
// Project gradients + values to the left face, from the right
//   returned values are on the i - 1 edges
//     (between i-1 and i cell center)
// ---------------------------------------------------------

arma_mat project_from_right(arma_mat values,
                            arma_mat gradients,
                            arma_mat x_centers,
                            arma_mat x_edges,
                            int64_t nGCs)
{
  int64_t nX = values.n_rows;
  int64_t nY = values.n_cols;

  // Define at edges:
  arma_mat projected(nX + 1, nY);
  projected.zeros();

  // no gradient in the 0 or iEnd cells
  for (int64_t j = 0; j < nY; j++)
  {
    for (int64_t i = 1; i < nX - 1; i++)
    {
      projected(i, j) = values(i, j) +
                        gradients(i, j) * (x_edges(i, j) - x_centers(i, j));
    }
    projected(0, j) = projected(1, j);
    projected(nX - 1, j) = projected(nX - 2, j);
    projected(nX, j) = projected(nX - 1, j);
  }
  return projected;
}

// ---------------------------------------------------------
// Limiter on values
//   projected is assumed to be on the edge between the
//   i-1 and i cell (i-1/2)
//   limited is returned at edges
// ---------------------------------------------------------

arma_vec limiter_value(arma_vec projected,
                       arma_vec values,
                       int64_t nPts,
                       int64_t nGCs)
{

  int64_t iStart = 0;
  int64_t iEnd = nPts + 2 * nGCs;

  arma_vec limited = projected;

  precision_t mini, maxi;

  for (int64_t i = iStart + 1; i < iEnd - 1; i++)
  {

    mini = values(i - 1);
    if (values(i) < mini)
      mini = values(i);
    maxi = values(i - 1);
    if (values(i) > maxi)
      maxi = values(i);

    if (limited(i) < mini)
      limited(i) = mini;
    if (limited(i) > maxi)
      limited(i) = maxi;
  }
  return limited;
}

// ---------------------------------------------------------
// take gradients and project to all edges
// ---------------------------------------------------------

projection_struct project_to_edges(arma_mat &values,
                                   arma_mat &x_centers, arma_mat &x_edges,
                                   arma_mat &y_centers, arma_mat &y_edges,
                                   int64_t nGCs)
{

  int64_t nX = values.n_rows;
  int64_t nY = values.n_cols;

  projection_struct proj;

  proj.gradLR = calc_grad(values, x_centers, nGCs, true);
  proj.gradDU = calc_grad(values.t(), y_centers.t(), nGCs, true).t();

  proj.R = project_from_left(values, proj.gradLR,
                             x_centers, x_edges, nGCs);
  // Left side of edge from left
  proj.L = project_from_right(values, proj.gradLR,
                              x_centers, x_edges, nGCs);
  // Up side of edge from down (left)
  proj.U = project_from_left(values.t(), proj.gradDU.t(),
                             y_centers.t(), y_edges.t(), nGCs)
               .t();
  // Down side of edge from up (right)
  proj.D = project_from_right(values.t(), proj.gradDU.t(),
                              y_centers.t(), y_edges.t(), nGCs)
               .t();

  return proj;
}

void Neutrals::solver_horizontal_cubesphere(Grid& grid, Times& time, Report& report) {
    // Function Reporting
    std::string function = "Neutrals::solver_horizontal_cubesphere";
    static int iFunction = -1;
    report.enter(function, iFunction);
    
    // Dimensions of Spatial Discretization
    int64_t nXs = grid.get_nX();
    int64_t nYs = grid.get_nY();
    int64_t nGCs = grid.get_nGCs();
    int64_t nAlts = grid.get_nAlts();
    int iAlt;

    // Time Discretization (TODO: change dt calculation method)
    precision_t dt = time.get_dt();

    // Advance for bulk calculation first, calculate for every altitude
    for (iAlt = 0; iAlt < nAlts; iAlt++) {
        /** Extract Grid Features **/
        arma_mat x = grid.refx_scgc.slice(iAlt);
        arma_mat xEdges = grid.refx_Left.slice(iAlt);
        arma_mat y = grid.refy_scgc.slice(iAlt);
        arma_mat yEdges = grid.refy_Down.slice(iAlt);

        // Get reference grid dimensions (Assume dx = dy and equidistant)
        arma_vec x_vec = x.col(0);
        precision_t dx = x_vec(1)-x_vec(0);
        precision_t area = dx * dx;
        arma_mat jacobian = grid.sqrt_g_scgc.slice(iAlt);

        /** States preprocessing **/
        /* MASS DENSITY */
        arma_mat rho = rho_scgc.slice(iAlt);

        /* VELOCITY */
        // Get spherical velocity
        arma_mat uVel = velocity_vcgc[0].slice(iAlt);
        arma_mat vVel = velocity_vcgc[1].slice(iAlt);
        // Convert to contravariant (reference) velocity
        arma_mat xVel = uVel % grid.A11_inv_scgc.slice(iAlt) + vVel % grid.A12_inv_scgc.slice(iAlt); // u^1
        arma_mat yVel = uVel % grid.A21_inv_scgc.slice(iAlt) + vVel % grid.A22_inv_scgc.slice(iAlt); // u^2
        // Generate contravriant momentum
        arma_mat xMomentum = rho % xVel; // x1momentum
        arma_mat yMomentum = rho % yVel; // x2momentum
        // Generate velocity magnitude squared
        arma_mat vel2 = xVel % xVel + yVel % yVel;

        /* TEMP and ENERGY */
        // Generate total energy (rhoE) (TODO: Verify)
        arma_mat rhoE = rho % (temperature_scgc.slice(iAlt) + 0.5*vel2);

        /** Advancing **/
        /* Initialize projection constructs and storages */
        projection_struct rhoP;
        projection_struct xMomentumP;
        projection_struct yMomentumP;
        projection_struct rhoEP;

        // They are all pure scalar fields without sqrt(g)
        arma_mat rhoL, rhoR, rhoD, rhoU;
        arma_mat xVelL, xVelR, xVelD, xVelU;
        arma_mat yVelL, yVelR, yVelD, yVelU;
        arma_mat totaleL, totaleR, totaleD, totaleU;

        arma_mat velL2, velR2, velD2, velU2;
        arma_mat tempL, tempR, tempD, tempU;
        arma_mat pressureL, pressureR, pressureD, pressureU;
        
        /** Initialize Flux and Wave Speed Storages */
        arma_mat eq1FluxLR, eq1FluxDU;
        arma_mat eq1FluxL, eq1FluxR, eq1FluxD, eq1FluxU;

        arma_mat eq2FluxLR, eq2FluxDU;
        arma_mat eq2FluxL, eq2FluxR, eq2FluxD, eq2FluxU;

        arma_mat eq3FluxLR, eq3FluxDU;
        arma_mat eq3FluxL, eq3FluxR, eq3FluxD, eq3FluxU;

        arma_mat eq4FluxLR, eq4FluxDU;
        arma_mat eq4FluxL, eq4FluxR, eq4FluxD, eq4FluxU;

        arma_mat wsL, wsR, wsD, wsU, wsLR, wsDU;

        arma_mat diff; // for Riemann Solver

        /* Projection */
        rhoP = project_to_edges(rho, x, xEdges, y, yEdges, nGCs);
        xMomentumP = project_to_edges(xMomentum, x, xEdges, y, yEdges, nGCs);
        yMomentumP = project_to_edges(yMomentum, x, xEdges, y, yEdges, nGCs);
        rhoEP = project_to_edges(rhoE, x, xEdges, y, yEdges, nGCs);

        // Resolve Scalar Fields into rho, xVel, yVel, and totalE (without rho)
        rhoL = rhoP.L;
        rhoR = rhoP.R;
        rhoD = rhoP.D;
        rhoU = rhoP.U;

        xVelL = xMomentumP.L / rhoL;
        xVelR = xMomentumP.R / rhoR;
        xVelD = xMomentumP.D / rhoD;
        xVelU = xMomentumP.U / rhoU;

        yVelL = yMomentumP.L / rhoL;
        yVelR = yMomentumP.R / rhoR;
        yVelD = yMomentumP.D / rhoD;
        yVelU = yMomentumP.U / rhoU;

        totaleL = rhoEP.L / rhoL;
        totaleR = rhoEP.R / rhoR;
        totaleD = rhoEP.D / rhoD;
        totaleU = rhoEP.U / rhoU;

        velL2 = xVelL % xVelL + yVelL % yVelL;
        velR2 = xVelR % xVelR + yVelR % yVelR;
        velD2 = xVelD % xVelD + yVelD % yVelD;
        velU2 = xVelU % xVelU + yVelU % yVelU;

        tempL = totaleL - 0.5 * velL2;
        tempR = totaleR - 0.5 * velR2;
        tempD = totaleD - 0.5 * velD2;
        tempU = totaleU - 0.5 * velU2;

        pressureL = (gamma_scgc.slice(iAlt) - 1) % (rhoP.L % tempL);
        pressureR = (gamma_scgc.slice(iAlt) - 1) % (rhoP.R % tempR);
        pressureD = (gamma_scgc.slice(iAlt) - 1) % (rhoP.D % tempD);
        pressureU = (gamma_scgc.slice(iAlt) - 1) % (rhoP.U % tempU);

        /* Calculate Edge Fluxes */
        // Note that dot product between normal vector at edge and flux vector
        // resolves into a pure one component flux or either hat{x} or hat{y}

        // Flux calculated from the left of the edge
        eq1FluxL = rhoL % xVelL % grid.sqrt_g_Left.slice(iAlt);
        // Flux calculated from the right of the edge
        eq1FluxR = rhoR % xVelR % grid.sqrt_g_Left.slice(iAlt);
        // Flux calculated from the down of the edge
        eq1FluxD = rhoD % yVelD % grid.sqrt_g_Down.slice(iAlt);
        // Flux calculated from the up of the edge
        eq1FluxU = rhoU % yVelU % grid.sqrt_g_Down.slice(iAlt);

        eq2FluxL = (rhoL % xVelL % xVelL + pressureL % grid.g11_upper_Left.slice(iAlt)) % grid.sqrt_g_Left.slice(iAlt);
        eq2FluxR = (rhoR % xVelR % xVelR + pressureR % grid.g11_upper_Left.slice(iAlt)) % grid.sqrt_g_Left.slice(iAlt);
        eq2FluxD = (rhoD % yVelD % xVelD + pressureD % grid.g12_upper_Down.slice(iAlt)) % grid.sqrt_g_Down.slice(iAlt);
        eq2FluxU = (rhoU % yVelU % xVelU + pressureU % grid.g12_upper_Down.slice(iAlt)) % grid.sqrt_g_Down.slice(iAlt);

        eq3FluxL = (rhoL % xVelL % yVelL + pressureL % grid.g21_upper_Left.slice(iAlt)) % grid.sqrt_g_Left.slice(iAlt);
        eq3FluxR = (rhoR % xVelR % yVelR + pressureR % grid.g21_upper_Left.slice(iAlt)) % grid.sqrt_g_Left.slice(iAlt);
        eq3FluxD = (rhoD % yVelD % yVelD + pressureD % grid.g22_upper_Down.slice(iAlt)) % grid.sqrt_g_Down.slice(iAlt);
        eq3FluxU = (rhoU % yVelU % yVelU + pressureU % grid.g22_upper_Down.slice(iAlt)) % grid.sqrt_g_Down.slice(iAlt);

        eq4FluxL = (totaleL % rhoL + pressureL) % xVelL % grid.sqrt_g_Left.slice(iAlt);
        eq4FluxR = (totaleR % rhoR + pressureR) % xVelR % grid.sqrt_g_Left.slice(iAlt);
        eq4FluxD = (totaleD % rhoD + pressureD) % yVelD % grid.sqrt_g_Down.slice(iAlt);
        eq4FluxU = (totaleU % rhoU + pressureU) % yVelU % grid.sqrt_g_Down.slice(iAlt);

        /* Wave Speed Calculation */
        wsL = sqrt(velL2) + sqrt(gamma_scgc.slice(iAlt) * (gamma_scgc.slice(iAlt) - 1) * tempL);
        wsR = sqrt(velR2) + sqrt(gamma_scgc.slice(iAlt) * (gamma_scgc.slice(iAlt) - 1) * tempR);
        wsD = sqrt(velD2) + sqrt(gamma_scgc.slice(iAlt) * (gamma_scgc.slice(iAlt) - 1) * tempD);
        wsU = sqrt(velU2) + sqrt(gamma_scgc.slice(iAlt) * (gamma_scgc.slice(iAlt) - 1) * tempU);

        wsLR = wsR;
        for (int i = 0; i < nXs + 1; i++)
        {
            for (int j = 0; j < nYs; j++)
            {
                if (wsL(i, j) > wsLR(i, j))
                wsLR(i, j) = wsL(i, j);
            }
        }

        wsDU = wsD;
        for (int i = 0; i < nXs; i++)
        {
            for (int j = 0; j < nYs + 1; j++)
            {
                if (wsU(i, j) > wsDU(i, j))
                wsDU(i, j) = wsU(i, j);
            }
        }

        /* Calculate average flux at the edges (Rusanov Flux) */
        diff = (rhoR - rhoL) % grid.sqrt_g_Left.slice(iAlt); // State difference, need to add sqrt(g)
        eq1FluxLR = (eq1FluxL + eq1FluxR) / 2 + 0.5 * wsLR % diff;
        diff = (rhoU - rhoD) % grid.sqrt_g_Down.slice(iAlt);
        eq1FluxDU = (eq1FluxD + eq1FluxU) / 2 + 0.5 * wsDU % diff;

        diff = (rhoR % xVelR - rhoL % xVelL) % grid.sqrt_g_Left.slice(iAlt);
        eq2FluxLR = (eq2FluxL + eq2FluxR) / 2 + 0.5 * wsLR % diff;
        diff = (rhoU % xVelU - rhoD % xVelD) % grid.sqrt_g_Down.slice(iAlt);
        eq2FluxDU = (eq2FluxD + eq2FluxU) / 2 + 0.5 * wsDU % diff;

        diff = (rhoR % yVelR - rhoL % yVelL) % grid.sqrt_g_Left.slice(iAlt);
        eq3FluxLR = (eq3FluxL + eq3FluxR) / 2 + 0.5 * wsLR % diff;
        diff = (rhoU % yVelU - rhoD % yVelD) % grid.sqrt_g_Down.slice(iAlt);
        eq3FluxDU = (eq3FluxD + eq3FluxU) / 2 + 0.5 * wsDU % diff;

        diff = (rhoR % totaleR - rhoL % totaleL) % grid.sqrt_g_Left.slice(iAlt);
        eq4FluxLR = (eq4FluxL + eq4FluxR) / 2 + 0.5 * wsLR % diff;
        diff = (rhoU % totaleU - rhoD % totaleD) % grid.sqrt_g_Down.slice(iAlt);
        eq4FluxDU = (eq4FluxD + eq4FluxU) / 2 + 0.5 * wsDU % diff;

        /* Update Bulk Scalars and Contravariant velocity */
        // Euler State Update
        for (int j = nGCs; j < nYs + nGCs; j++)
        {
            for (int i = nGCs; i < nXs + nGCs; i++)
            {
                precision_t rhoResidual_ij = dx * eq1FluxLR(i + 1, j) -
                                                dx * eq1FluxLR(i, j) +
                                                dx * eq1FluxDU(i, j + 1) -
                                                dx * eq1FluxDU(i, j);
                rho(i, j) = rho(i, j) - dt / area / jacobian(i, j) * rhoResidual_ij;
                precision_t xMomentumResidual_ij = dx * eq2FluxLR(i + 1, j) -
                                                    dx * eq2FluxLR(i, j) +
                                                    dx * eq2FluxDU(i, j + 1) -
                                                    dx * eq2FluxDU(i, j);
                xMomentum(i, j) = xMomentum(i, j) - dt / area / jacobian(i, j) * xMomentumResidual_ij;
                precision_t yMomentumResidual_ij = dx * eq3FluxLR(i + 1, j) -
                                                    dx * eq3FluxLR(i, j) +
                                                    dx * eq3FluxDU(i, j + 1) -
                                                    dx * eq3FluxDU(i, j);
                yMomentum(i, j) = yMomentum(i, j) - dt / area / jacobian(i, j) * yMomentumResidual_ij;
                precision_t rhoEResidual_ij = dx * eq4FluxLR(i + 1, j) -
                                                dx * eq4FluxLR(i, j) +
                                                dx * eq4FluxDU(i, j + 1) -
                                                dx * eq4FluxDU(i, j);
                rhoE(i, j) = rhoE(i,j) - dt / area / jacobian(i, j) * rhoEResidual_ij;
            }
        }

        /* Re-derive Spherical Velocity and Bulk States */
        // Density
        rho_scgc.slice(iAlt) = rho;

        // Bulk Velocity
        xVel = xMomentum / rho; // u^1
        yVel = yMomentum / rho; // u^2
        vel2 = xVel % xVel + yVel % yVel; // Squared Magnitude of Contravariant 
        velocity_vcgc[0].slice(iAlt) = xVel%grid.A11_scgc.slice(iAlt) + yVel%grid.A12_scgc.slice(iAlt);
        velocity_vcgc[1].slice(iAlt) = xVel%grid.A21_scgc.slice(iAlt) + yVel%grid.A22_scgc.slice(iAlt);

        // Bulk Temperature
        temperature_scgc.slice(iAlt) = rhoE / rho - 0.5*vel2;

        /* Update specie density */
        

        report.exit(function);
        return;
    }

}