// Copyright 2023, the Aether Development Team (see doc/dev_team.md for members)
// Full license can be found in License.md

// Initial version: F. Cheng, July 2023

#include "aether.h"

using namespace Cubesphere_tools;

void Neutrals::solver_horizontal_cubesphere_advection(Grid& grid, Times& time) {
    // Function Reporting
    std::string function = "Neutrals::solver_horizontal_cubesphere_advection";
    static int iFunction = -1;
    report.enter(function, iFunction);
    
    // Dimensions of Spatial Discretization
    int64_t nXs = grid.get_nX();
    int64_t nYs = grid.get_nY();
    int64_t nGCs = grid.get_nGCs();
    int64_t nAlts = grid.get_nAlts();
    int iAlt, iSpec;

    // Time Discretization (TODO: change dt calculation method)
    precision_t dt = time.get_dt();


    // Advance for bulk calculation first, calculate for every altitude
    for (iAlt = nGCs; iAlt < nAlts-nGCs; iAlt++) {
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
        
        // Generate velocity magnitude squared
        arma_mat vel2 = xVel % xVel + yVel % yVel;

        /** Advancing **/
        /* Initialize projection constructs and storages */
        projection_struct rhoP;
        projection_struct xVelP;
        projection_struct yVelP;

        // They are all pure scalar fields without sqrt(g)
        arma_mat rhoL, rhoR, rhoD, rhoU;
        arma_mat xVelL, xVelR, xVelD, xVelU;
        arma_mat yVelL, yVelR, yVelD, yVelU;

        arma_mat velL2, velR2, velD2, velU2;

        /** Initialize Flux and Wave Speed Storages */
        arma_mat eq1FluxLR, eq1FluxDU;
        arma_mat eq1FluxL, eq1FluxR, eq1FluxD, eq1FluxU;

        arma_mat wsL, wsR, wsD, wsU, wsLR, wsDU;

        arma_mat diff; // for Riemann Solver

        /* Projection */
        rhoP = project_to_edges(rho, x, xEdges, y, yEdges, nGCs);
        xVelP = project_to_edges(xVel, x, xEdges, y, yEdges, nGCs);
        yVelP = project_to_edges(yVel, x, xEdges, y, yEdges, nGCs);

        // Resolve Scalar Fields into rho, xVel, yVel, and totalE (without rho)
        rhoL = rhoP.L;
        rhoR = rhoP.R;
        rhoD = rhoP.D;
        rhoU = rhoP.U;

        xVelL = xVelP.L;
        xVelR = xVelP.R;
        xVelD = xVelP.D;
        xVelU = xVelP.U;

        yVelL = yVelP.L;
        yVelR = yVelP.R;
        yVelD = yVelP.D;
        yVelU = yVelP.U;

        velL2 = xVelL % xVelL + yVelL % yVelL;
        velR2 = xVelR % xVelR + yVelR % yVelR;
        velD2 = xVelD % xVelD + yVelD % yVelD;
        velU2 = xVelU % xVelU + yVelU % yVelU;


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

        /* Wave Speed Calculation */
        wsL = sqrt(velL2);
        wsR = sqrt(velR2);
        wsD = sqrt(velD2);
        wsU = sqrt(velU2);

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

        /* Update Bulk Scalars and Contravariant velocity */
        // Euler State Update
        for (int j = nGCs; j < nYs - nGCs; j++)
        {
            for (int i = nGCs; i < nXs - nGCs; i++)
            {
                precision_t rhoResidual_ij = dx * eq1FluxLR(i + 1, j) -
                                                dx * eq1FluxLR(i, j) +
                                                dx * eq1FluxDU(i, j + 1) -
                                                dx * eq1FluxDU(i, j);
                rho(i, j) = rho(i, j) - dt / area / jacobian(i, j) * rhoResidual_ij;
            }
        }

        /* Re-derive Spherical Velocity and Bulk States */
        // Density
        rho_scgc.slice(iAlt) = rho;

        // Bulk Velocity
        //vel2 = xVel % xVel + yVel % yVel; // Squared Magnitude of Contravariant 
        //velocity_vcgc[0].slice(iAlt) = xVel%grid.A11_scgc.slice(iAlt) + yVel%grid.A12_scgc.slice(iAlt);
        //velocity_vcgc[1].slice(iAlt) = xVel%grid.A21_scgc.slice(iAlt) + yVel%grid.A22_scgc.slice(iAlt);

        /* Update specie density */
        for (iSpec = 0; iSpec < nSpecies; iSpec++) {
            //species[iSpec].density_scgc.slice(iAlt) = rho % species[iSpec].concentration_scgc.slice(iAlt);
            species[iSpec].density_scgc.slice(iAlt) = rho/species[iSpec].mass;
        }


        report.exit(function);
        return;
    }
}