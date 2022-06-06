postesSynop.csv :

  * contains weather stations id, localisations and names :
  * variables
    - ID : weather station ID (used to identify the station in the other weather data file)
    - Nom : name of the location
    - Latitude/Longitude : coordinates
    - Altitude : altitude above sea level


france.parquet :

  * contains weather data from 2017/09/01 to 2022/06/03 for the metropolitan weather stations
  * variables :
    - date : corresponding calendar date of wather measurement
    - rr24 : rainfall in the past 24 hours
    - t : raw temperature (in °K) (not reprensetative of the overall daily temperature)
    - t_max : maximal temperature (in °C) at a given date
    - t_min : minimal temperature (in °C) at a given date
    - DJ_X : "degre-jour" temperature (in °C) for base X. This temperature is used in agro-science to quantify the amount of heat useful to the crop. The corresponding formula is : DJ_X = clip((t_max + t_min)/2 - X, 0, 30)
    -cumul_DJ_X : cumulated version of DJ_X from 2017/09/01, useful for phenological analysis of crops, cumulative sum can start from another date by subtracting the first date value from the time serie.
    - id_sta : weather station ID


franceagrimer-rdts-surfs-multicrops.parquet:

  * contains crop yields, crop surface, and crop production from 2017 to 2021
  * variables :
    - n_dep : french administration departement/region number (NB: above 95, rows are related to regions and can be discarded)
    - dep : departement name
    - surf_X : cultivated crop surface for year X (in ha)
    - rdt_X : crop yield for year X (in 100kg/ha)
    - prod_X : crop production for year X (in t)
    - crop : name of the crop :
      	   - MA : maïze
	   - BTH : winter soft wheat
	   - BTP : sprint soft wheat
	   - BDH : winter durum wheat
	   - BDP : spring durum wheat
	   - OH : winter barley
	   - OP : spring barley
	   - CZH : rapeseed
	   - TS : sunflower
