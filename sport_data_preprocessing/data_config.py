action_all = ['assist',
              'block',
              'carry',
              'check',
              'controlledbreakout',
              'controlledentryagainst',
              'dumpin',
              'dumpinagainst',
              'dumpout',
              'faceoff',
              'goal',
              'icing',
              'lpr',
              'offside',
              'pass',
              'pass1timer',
              'penalty',
              'pressure',
              'pscarry',
              'pscheck',
              'pslpr',
              'pspuckprotection',
              'puckprotection',
              'reception',
              'receptionprevention',
              'shot',
              'shot1timer',
              # 'socarry',
              # 'socheck',
              # 'sogoal',
              # 'solpr',
              # 'sopuckprotection',
              # 'soshot'
              ]

interested_raw_features = [ 'xAdjCoord',
                            'yAdjCoord',
                            'scoreDifferential',
                            'manpowerSituation', #powerPlay: 1, evenStrength: 0, shortHanded: -1
                            'outcome', #successful: 1, undetermined: 0, failed : -1
                           ]

interested_compute_features = ['velocity_x',
                               'velocity_y',
                               'time_remain',
                               'duration',
                               'home_away',
                               'angle2gate'
                               ]
teamList = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 322] # 31 team

positions = ['C', 'RW', 'LW', 'D', 'G']

