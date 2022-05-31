  def cultured_neurons(self, extra, fi_ranges, label, channel_name, channel, update, isWT):
        # channel_name, channel, dict_fn, wt_mul, mut_mul, name
        # 'na16mut', 'na16', './params/na16_mutv2.txt', 2+extra, 0, 'WT'
        dict_fn, const, name = update
        if isWT:
            self.update_channel(channel_name, channel, dict_fn, extra, const)
        else:
            self.update_channel(channel_name, channel, dict_fn, const, extra)
        #self.plot_stim(0.5,f'{label}_{name}_{extra}') # todo uncomment
        x_axis, npeaks = self.get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        return [x_axis, npeaks, name]

    def explore_param(self, ch_name, gbar_name, ranges, extras, fi_ranges, channel_name, channel, updates, isTTX=False):
        all_prevs = []
        all_FIs = []
        for i in range(len(ranges)):
            self.__init__()
            curr_factor = ranges[i]
            extra = np.round(extras[i], 2)
            prev = self.update_K(ch_name, gbar_name, curr_factor)
            label = f'{ch_name}_{curr_factor}'
            fi_range_wt = fi_ranges[0]
            fi_range_mut = fi_ranges[1]
            update_wt = updates[0]
            update_mut = updates[1]
            #fi_wt = cultured_neurons_wt(extra, [0.1, 1, 5],label)  # set endpoint from 5 to 6 for even spacing across types
            #fi_mut = cultured_neurons_mut(extra, [0.1, 1, 5],label)  # set endpoint from 5 to 6 for even spacing across types
            fi_wt = self.cultured_neurons(extra+2, fi_range_wt, label, channel_name, channel, update_wt, isWT=True)
            fi_mut = self.cultured_neurons(extra, fi_range_mut, label, channel_name, channel, update_mut, isWT=False)
            if isTTX:
                fi_range_wtTTX = fi_ranges[2]
                fi_range_mutTTX = fi_ranges[3]
                update_wtTTX = updates[2]
                update_mutTTX = updates[3]
                #fi_wtTTX = cultured_neurons_wtTTX(extra, [0.4, 2, 5], label)
                #fi_mutTTX = cultured_neurons_mutTTX(extra, [0.4, 2, 5], label)
                fi_wtTTX = self.cultured_neurons(extra, fi_range_wtTTX, label, channel_name, channel, update_wtTTX, isWT=True)
                fi_mutTTX = self.cultured_neurons(extra, fi_range_mutTTX, label, channel_name, channel, update_mutTTX, isWT=False)
                all_FIs.append([fi_wt, fi_mut, fi_wtTTX, fi_mutTTX])
            else:
                all_FIs.append([fi_wt, fi_mut])
            all_prevs.append(prev)

            # plot FIs
            #self.plot_all_FIs(all_FIs[i], i, isTTX)  # running with the decided paramters (gSKV3.1x2 and extra of 0.1)

            # revert h back to initial condition
            self.reverse_update_K(ch_name, gbar_name, all_prevs[0])

        return all_FIs
   ### Other Functions ###

    def cultured_neurons_wt(self, extra,fi_ranges,label):
        update_na16('./params/na16_mutv2.txt',2+extra,0)
        plot_stim(0.5,f'{label}_overexpressedWT{extra}')
        #make_fi(fi_ranges,f'{label}_overexpressedWT{extra}')
        #name = f'{label} WT {extra}' # label is ch_name and condition
        name = f'WT'
        x_axis, npeaks = get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        #fig.savefig(f'./Plots/Kexplore/{fn}_FI.pdf')
        return [x_axis, npeaks, name]

    def cultured_neurons_mut(self, extra,fi_ranges,label):
        update_na16('./params/na16_mutv2.txt',2,extra)
        plot_stim(0.5,f'{label}_overexpressedMut{extra}')
        #make_fi(fi_ranges,f'{label}_overexpressedMut{extra}')
        #name = f'{label} Mut {extra}'
        name = f'Mut'
        x_axis, npeaks = get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        return [x_axis, npeaks, name]

    def cultured_neurons_wtTTX(self, extra,fi_ranges,label):
        update_na16('./params/na16_mutv2.txt',extra,0)
        plot_stim(2,f'{label}_overexpressedWT_TTX{extra}')
        #make_fi(fi_ranges,f'{label}_overexpressedWT_TTX{extra}')
        #name = f'{label} WT_TTX {extra}'
        name = f'WT_TTX'
        x_axis, npeaks = get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        return [x_axis, npeaks, name]

    def cultured_neurons_mutTTX(self, extra,fi_ranges,label):
        update_na16('./params/na16_mutv2.txt',0,extra)
        plot_stim(2,f'{label}_overexpressedmut_TTX{extra}')
        #make_fi(fi_ranges,f'{label}_overexpressedmut_TTX{extra}')
        #name = f'{label} mut_TTX {extra}'
        name = f'mut_TTX'
        x_axis, npeaks = get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        return [x_axis, npeaks, name]


    def plot_wt_vs_ttx(self, fi_arr):
        init_settings()
        extra = 0.1
        k_factor = 2
        prev = update_K('SKv3_1','gSKv3_1bar', k_factor)
        label = f'FI_curves'
        fi_wt = cultured_neurons_wt(extra, fi_arr,label)
        fi_mut = cultured_neurons_mut(extra, fi_arr,label)
        fi_wtTTX = cultured_neurons_wtTTX(extra, fi_arr, label)
        fi_mutTTX = cultured_neurons_mutTTX(extra, fi_arr, label)
        all_FIs = [fi_wt, fi_mut, fi_wtTTX, fi_mutTTX]
        plot_all_FIs(all_FIs)

    def plot_wt_vs_het(self, fi_ranges):
        init_settings()
        k_factor = 2
        prev = update_K('SKv3_1','gSKv3_1bar', k_factor)
        label = f'WT_500pA'
        #NO MUT AND 2 ALLELS OF WT
        update_na16('./params/na16_mutv2.txt',2,0)
        plot_stim(0.5,label,clr='black')
        wt_x_axis, wt_npeaks = get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        reverse_update_K('SKv3_1','gSKv3_1bar', prev)
        init_settings()
        prev = update_K('SKv3_1','gSKv3_1bar', k_factor)
        update_na16('./params/na16_mutv2.txt',1,1)
        label = f'Het_500pA'
        plot_stim(0.5,label,clr='red')
        het_x_axis, het_npeaks = get_fi_curve(fi_ranges[0], fi_ranges[1], fi_ranges[2])
        fig = plt.figure()
        plt.plot(wt_x_axis, wt_npeaks, label='WT', color='black')
        plt.plot(het_x_axis, het_npeaks, label='HET', color='red')
        filename= f'./Plots/V1/FI_Het_plots.pdf'
        fig.savefig(filename)

    def het_sims(self):
        init_settings()
        update_na16('./params/na16_mutv2.txt',1,1)
        init_stim(amp=0.5)
        Vm, I, t, stim = run_model()
        plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/hetrozygous_500pA',times=t,color_str='blue')


  if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='Generate simulated data.')
      parser.add_argument("--function", "-f", type=int, default=0, help="Specify which function to run")
      # plot simulation for cultured neurons with and without ttx
      args = parser.parse_args()
      if args.function == 0:
          plot_wt_vs_ttx([0, 1.8, 10])
      # plot simulation for WT and Hetrozygous
      if args.function == 1:
          plot_wt_vs_het([0, 1.8, 10])

      if args.function == 2:
          # gK_Tstbar_K_Tst
          # gK_Pstbar_
          # update_K('SKv3_1','gSKv3_1bar',2)
          # update_K('K_Tst','gK_Tstbar',2)
          # update_K('K_Pst','gK_Pstbar',2)
          # cultured_neurons_mut(0.25,[0.1, 1, 5])
          # cultured_neurons_wt(0.5,[0.1, 1, 5])
          # cultured_neurons_wtTTX(0.5,[0.4, 2, 6])
          # cultured_neurons_mutTTX(0.25,[0.4, 2, 6])

          # run 3
          conditions = np.repeat(2, 5)  # factors
          extras = np.arange(0.1, 0.6, 0.1)
          explore_param('SKv3_1', 'gSKv3_1bar', conditions, extras)
          # explore_param('K_Tst','gK_Tstbar', [1, 10, 100])
          # explore_param('K_Pst','gK_Pstbar', [1, 2, 3])

      if args.function == 3:
          # run 1
          extras = np.repeat(0.25, 3)  # for mut channel # HOLD WT AT 0.5 (need to adjust code for this)
          explore_param('SKv3_1', 'gSKv3_1bar', [1, 2, 3], extras)
          explore_param('K_Tst', 'gK_Tstbar', [1, 10, 100], extras)
          explore_param('K_Pst', 'gK_Pstbar', [1, 2, 3], extras)

      if args.function == 4:
          # run 2
          init_settings()

          # create mutTTX
          init_stim(amp=0.5)
          update_na16('./params/na16_mutv2.txt', 0, 0.25)
          update_K('SKv3_1', 'gSKv3_1bar', 2)
          I = plot_stim(2, f'mut_overexpressedmut_TTX_0.25')
          fig, ax1 = plt.subplots(1, 1)
          ax1.plot(I['Na'], color='black')
          ax1.plot(I['K'], color='blue')
          ax1.plot(I['Ca'], color='green')
          fig.savefig('./Plots/mut_vclamp.pdf')

      if args.function == 5:
          ### new format ###

          # create model
          model = NeuronModel()
          isTTX = True
          # explore param
          ch_name = 'SKv3_1'
          gbar_name = 'gSKv3_1bar'
          ranges = np.repeat(0.25, 3)  # conditions / factors
          extra = np.arange(0.1, 0.6, 0.1)  # [1, 2, 3]

          # cultured_neurons
          fi_ranges = [[0.1, 1, 5], [0.1, 1, 5], [0.4, 2, 5], [0.4, 2, 5]]  # wt, mut, wtttx, mutttx

          # update_channel
          # dict_fn, wt_mul or mut_mul=const, name for each wt, mut, wtttx, mutttx
          updates = [['./params/na16_mutv2.txt', 0, 'WT'], ['./params/na16_mutv2.txt', 2, 'MUT'],
                     ['./params/na16_mutv2.txt', 0, 'WtTTX'], ['./params/na16_mutv2.txt', 0, 'MutTTX']]
          channel_name = 'na16mut'
          channel = 'na16'

          # execute
          condition_data = model.explore_param(ch_name, gbar_name, ranges, extra,
                                               fi_ranges,
                                               channel_name, channel, updates, isTTX)
          model.plot_all_FIs(condition_data, isTTX)

