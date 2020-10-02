/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/
#ifndef __GRID_CONSTANTS_H
#define __GRID_CONSTANTS_H
enum func_ {

    GRID_FUNC_AB = 100,
    GRID_FUNC_DADB = 200,
    GRID_FUNC_ADBmDAB_X = 301,
    GRID_FUNC_ADBmDAB_Y = 302,
    GRID_FUNC_ADBmDAB_Z = 303,
    GRID_FUNC_ARDBmDARB_XX = 411,
    GRID_FUNC_ARDBmDARB_XY = 412,
    GRID_FUNC_ARDBmDARB_XZ = 413,
    GRID_FUNC_ARDBmDARB_YX = 421,
    GRID_FUNC_ARDBmDARB_YY = 422,
    GRID_FUNC_ARDBmDARB_YZ = 423,
    GRID_FUNC_ARDBmDARB_ZX = 431,
    GRID_FUNC_ARDBmDARB_ZY = 432,
    GRID_FUNC_ARDBmDARB_ZZ = 433,
    GRID_FUNC_DABpADB_X = 501,
    GRID_FUNC_DABpADB_Y = 502,
    GRID_FUNC_DABpADB_Z = 503,
    GRID_FUNC_DX = 601,
    GRID_FUNC_DY = 602,
    GRID_FUNC_DZ = 603,
    GRID_FUNC_DXDY = 701,
    GRID_FUNC_DYDZ = 702,
    GRID_FUNC_DZDX = 703,
    GRID_FUNC_DXDX = 801,
    GRID_FUNC_DYDY = 802,
    GRID_FUNC_DZDZ = 803
};

#define GRID_BACKEND_AUTO 10
#define GRID_BACKEND_REF 11
#define GRID_BACKEND_DGEMM 12
#define GRID_BACKEND_GPU 13

#endif
