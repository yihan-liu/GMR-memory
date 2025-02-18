# key_frames.py

TS_T_S = [1764996]                                          # TS: triangle start
TS_T_E = [1839995]                                          # TS: triangle end
TS_S_S = [105996, 3539995]                                  # TS: square start
TS_S_E = [164996, 3634994]                                  # TS: square end

SC_S_S = [195567, 786462, 1219327]                          # SC: square start
SC_S_E = [275514, 850106, 1276297]                          # SC: square end
SC_C_S = [8698, 912093, 1438514]                            # SC: circle start
SC_C_E = [68177, 955910, 1496824]                           # SC: circle end

SQ_S   = [83999, 485999, 767999, 1165999]                   # SQ: square start
SQ_E   = [135999, 545999, 845999, 1237999]                  # SQ: square end

CR_S   = [27999, 310999, 540999, 825999]                    # CR: circle start
CR_E   = [117999, 394999, 594999, 899999]                   # CR: circle end

TR_S   = [1, 221999, 601999, 879999, 1141999, 1519999]      # TR: triangle start
TR_E   = [57999, 291999, 671999, 959999, 1245999, 1619999]  # TR: triangle end

TS_intervals = {
    0: list(zip(TS_T_S, TS_T_E)),       # Triangle intervals
    1: list(zip(TS_S_S, TS_S_E)),       # Square intervals
    2: []                               # No circle intervals
}

SC_intervals = {
    0: [],                              # No triangle intervals
    1: list(zip(SC_S_S, SC_S_E)),       # Square intervals
    2: list(zip(SC_C_S, SC_C_E))        # Circle intervals
}

SQ_intervals = {
    0: [],                              # No triangle intervals
    1: list(zip(SQ_S, SQ_E)),           # Square intervals
    2: []                               # No circle intervals
}


CR_intervals = {
    0: [],                              # No triangle intervals
    1: [],                              # No square intervals
    2: list(zip(CR_S, CR_E))            # Circle intervals
}

TR_intervals = {
    0: list(zip(TR_S, TR_E)),           # Triangle intervals
    1: [],                              # No square intervals
    2: []                               # No circle intervals
}