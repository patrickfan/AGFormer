import os
import numpy as np

def intp_clatlon_bilinear(clat,clon,lat,lon,data2d):
#################################################################
#
# Bilinear interpolation of structured [lat,lon] data to
# a target point [clat,clon]
#
#  - Data must be on an equirectangular lat-lon grid.
#
#  - Lat, Lon coordinates in an increasing order with 1-dimension
#    (South->North), (West->East)
#    lat.shape -> nlat; lon.shape -> nlon
#
#  - Shape of data2d = [nlat, nlon]
#
#                                                     by H. Kang
#################################################################

    # Lat,lon to I,J ========================================

    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]
    slon = lon[0]
    slat = lat[0]
    nlon = lon.shape[0]
    nlat = lat.shape[0]


    if ( clon < 0.0 ):
         clon = clon+360.0
    if ( slon < 0.0 ):
         slon = slon+360.0

    #-------------------------------------------------------
    # A schematic of a grid box with a target point
    #
    #    (j_ul,i_ul)        (j_ur,i_ur)
    #               X-------X
    #               |       |
    #               |  *    |
    #               |       |
    #               X-------X
    #    (j_ll,i_ll)        (j_lr,i_lr)
    #
    #                       * = Target point at (clat,clon)
    #-------------------------------------------------------

    # Convert clon,clat to indices of a lower left corner
    j_ll = int((clat-slat)/dlat)
    i_ll = int((clon-slon)/dlon)

    # Lower right
    j_lr = j_ll
    i_lr = i_ll + 1

    # Upper left
    j_ul = j_ll + 1
    i_ul = i_ll

    # Upper right
    j_ur = j_ll + 1
    i_ur = i_ll + 1

    #========================================================

    lon_ll = lon[i_ll]
    lon_lr = lon[i_lr]

    if ( lon_ll < 0.0 ):
         lon_ll = lon_ll + 360
    if ( lon_lr < 0.0 ):
         lon_lr = lon_lr + 360
    #========================================================

    # Bilinear interpolation weights
    wgt_lon_r = (clon - lon_ll) / dlon
    wgt_lon_l = (lon_lr - clon) / dlon
    wgt_lat_u = (clat - lat[j_ll]) / dlat
    wgt_lat_l = (lat[j_ul] - clat) / dlat

    # Lon interpolation first
    cdata_u = ( data2d[j_ul,i_ul] * wgt_lon_l +
                data2d[j_ur,i_ur] * wgt_lon_r  )
    cdata_l = ( data2d[j_ll,i_ll] * wgt_lon_l +
                data2d[j_lr,i_lr] * wgt_lon_r  )

    # Lat interpolation
    cdata = cdata_u * wgt_lat_u + cdata_l * wgt_lat_l

    return cdata

#============================================================

if __name__ == "__main__":

     data_name = '/lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/train/2017_9.npz'
     data = np.load(data_name)

     # Lat & Lon setup --> Only works for 0.25 deg lat-lon data like ERA5
     ddeg = 0.25
     lat = data["latitude"][0,0,:,0]

     nlon = int(360/ddeg)
     lon = np.zeros(nlon)

     for i in range(nlon):
          lon[i] = i * ddeg

     var_name = "2m_temperature"
     var = data[var_name]
     nt = var.shape[0]  # Number of time frames
     var_cdata = np.zeros(nt)

     # clat, clon (BSR)
     clat = 42.24923291
     clon = -109.4286346

     # Time loop
     for n in range(nt):
          var_cdata[n] = intp_clatlon_bilinear(clat,clon,lat,lon,var[n,0,:,:])
          print(n,var_cdata[n])
