// Copyright 2023, the Aether Development Team (see doc/dev_team.md for members)
// Full license can be found in License.md
//
// initial version - A. Ridley - July 28, 2023

#include "aether.h"

bool Neutrals::advect_horizontal(Grid & grid, Times & time) {
    bool didWork = true;

    std::string function = "Neutrals::advance_horizontal";
    static int iFunction = -1;
    report.enter(function, iFunction);
 
    solver_horizontal_cubesphere(grid, time);

    report.exit(function);
    return didWork;
}