import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import glob
import numpy as np
import vtk
from vtk_utils.vtk_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from matplotlib.legend_handler import HandlerTuple

def plot_dice_scores_as_boxplot(dir_n_list, method_list, ax):
    dfs = []
    #classes = ['WH', 'LV', 'RV', 'LA', 'RA', 'Myo', 'Ao', 'PA']
    #classes = ['Myo']
    classes = ['WH']
    sort_index = None
    df_m = []
    for dir_n, m in zip(dir_n_list, method_list):
        data = np.genfromtxt(dir_n, delimiter=",", usemask=True, skip_header=1)[:, 0]
        #data = np.delete(data, 10, axis=0)
        print(data.shape)
        df = pd.DataFrame(data, columns=classes)
        df['Method'] = [m]*len(df.index)
        df_m.append(df)
    dfm_concat = pd.concat(df_m, axis=0)
    dfm_melt = dfm_concat.melt(id_vars = 'Method', value_vars=sort_index, var_name = 'Classes')
    print(dfm_melt)
    sns.set(style="whitegrid")
    #sns.boxplot(data=dfm_melt.reset_index(), linewidth=1., y='value', x='Classes',hue='Method', showmeans=True, dodge=False,\
    #        meanprops={"marker":"o",
    #               "markerfacecolor":"white",
    #               "markeredgecolor":"black",
    #              "markersize":"4"}, \
    #             ax=ax,flierprops = dict(markerfacecolor = '0.50', markersize = 2))
    print("""""""""""""")
    print(dfm_melt.reset_index())
    sns.boxplot(data=dfm_melt, y='value', x='Classes',hue='Method', showmeans=True, meanprops={'marker':'o',
                       'markerfacecolor':'white', 
                       'markeredgecolor':'black',
                       'markersize':'8'}, \
                        ax=ax,\
                        boxprops={'alpha': 0.4})

    ax.set_title('VSD Dice Scores', size=20)
    ax.set_ylim([-0.1,1])
    ax.set_title('WH Dice Scores', size=20)
    ax.set_ylim([0.6,1])
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax.tick_params(labelsize=16)
    ax.axes.xaxis.label.set_visible(False)
    plt.xticks(visible=False)
    ax.axes.yaxis.label.set_visible(False)
    sns.stripplot(data=dfm_melt, y='value', x='Classes',hue='Method',dodge=True, jitter=0.3, size=6,linewidth=1,ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[(handles[i], handles[i+len(method_list)]) for i in range(len(method_list))],\
          labels=labels[:len(method_list)],\
          loc='upper center', bbox_to_anchor=(0.5, -0.05), handlelength=4,ncol=3,
          handler_map={tuple: HandlerTuple(ndivide=None)})
    return

def plot_dice_scores_over_epochs(dir_n, ax, fn='dice.csv'):
    classes = ['WH', 'LV', 'RV', 'LA', 'RA', 'Myo', 'Ao', 'PA']
    classes = ['Myo', 'Myo']
    classes_to_show = [0]
    
    csv_fns = glob.glob(os.path.join(dir_n, '**', fn))
    # compute total dice scores first
    dice_list, dice_std_list, epoch_list = [], [], []
    for fn in csv_fns:
        epoch = int(os.path.basename(os.path.dirname(fn)).split('_')[-1])
        epoch_list.append(epoch)
        data = np.loadtxt(fn, delimiter=',', skiprows=1)
        # REMOVE 10-12 FOR UNET EXPERIMENT DUE TO MISMATCH 
        #data = np.delete(data, [10], axis=0)
        #if 'MUnetBase16_aligned' in dir_n:
        #    data = np.delete(data, [10, 11, 12], axis=0)
        #    print(data)
        mean_dice = np.mean(data, axis=0)
        std_dice = np.std(data, axis=0)
        dice_list.append(mean_dice)
        dice_std_list.append(std_dice)
    sortid = np.argsort(epoch_list)
    epoch_list = np.array([epoch_list[i] for i in sortid])
    dice_list = [dice_list[i] for i in sortid]
    dice_std_list = [dice_std_list[i] for i in sortid]
    dice_list = np.array(dice_list)
    dice_std_list = np.array(dice_std_list)
    for i in range(len(classes_to_show)):
        ax.errorbar(epoch_list + np.random.rand(), dice_list[:, i], yerr=dice_std_list[:, classes_to_show[i]], label=classes[classes_to_show[i]], linewidth=2, capsize=5)
    ax.legend(loc='lower right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_ylim([0.3, 1])
    return ax

def setupLuts(name='cool_to_warm'):
    luts = []
    if name == 'blue_to_red':
        # HSV (Blue to REd)  Default
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.SetNumberOfColors(256)
        lut.Build()
    elif name =='cool_to_warm':
        # Diverging (Cool to Warm) color scheme
        ctf = vtk.vtkColorTransferFunction()
        ctf.SetColorSpaceToDiverging()
        ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
        ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)
        cc = list()
        for i in range(256):
          cc.append(ctf.GetColor(float(i) / 255.0))
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        for i, item in enumerate(cc):
          lut.SetTableValue(i, item[0], item[1], item[2], 1.0)
        lut.Build()
    elif name =='shock':
        # Shock
        ctf = vtk.vtkColorTransferFunction()
        min = 93698.4
        max = 230532
        ctf.AddRGBPoint(self._normalize(min, max,  93698.4),  0.0,         0.0,      1.0)
        ctf.AddRGBPoint(self._normalize(min, max, 115592.0),  0.0,         0.905882, 1.0)
        ctf.AddRGBPoint(self._normalize(min, max, 138853.0),  0.0941176,   0.733333, 0.027451)
        ctf.AddRGBPoint(self._normalize(min, max, 159378.0),  1.0,         0.913725, 0.00784314)
        ctf.AddRGBPoint(self._normalize(min, max, 181272.0),  1.0,         0.180392, 0.239216)
        ctf.AddRGBPoint(self._normalize(min, max, 203165.0),  1.0,         0.701961, 0.960784)
        ctf.AddRGBPoint(self._normalize(min, max, 230532.0),  1.0,         1.0,      1.0)
        cc = list()
        for i in xrange(256):
          cc.append(ctf.GetColor(float(i) / 255.0))
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        for i, item in enumerate(cc):
          lut.SetTableValue(i, item[0], item[1], item[2], 1.0)
        lut.Build()
    elif name =='set3':
        colorSeries = vtk.vtkColorSeries()
        # Select a color scheme.
        #colorSeriesEnum = colorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_9
        # colorSeriesEnum = colorSeries.BREWER_DIVERGING_SPECTRAL_10
        # colorSeriesEnum = colorSeries.BREWER_DIVERGING_SPECTRAL_3
        # colorSeriesEnum = colorSeries.BREWER_DIVERGING_PURPLE_ORANGE_9
        # colorSeriesEnum = colorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_9
        # colorSeriesEnum = colorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_9
        colorSeriesEnum = colorSeries.BREWER_QUALITATIVE_SET3
        # colorSeriesEnum = colorSeries.CITRUS
        colorSeries.SetColorScheme(colorSeriesEnum)
        lut = vtk.vtkLookupTable()
        #lut.SetNumberOfColors(20)
        colorSeries.BuildLookupTable(lut, vtk.vtkColorSeries.CATEGORICAL)
        #colorSeries.BuildLookupTable(lut, vtk.vtkColorSeries.ORDINAL)
        lut.SetNanColor(1, 0, 0, 1)
    else:
        raise NotImplementedError('Requested color scheme not implemented.')
    return lut

def visualize_template_over_epochs(dir_n):
    pass

def visualize_template(mesh_fn, array_name='RegionId', threshold_name='RegionId', show_edge=False, id_to_visualize=-1, tilt_factor=0., digit_size=256, camera=None, gt=False):
    mesh = load_vtk_mesh(mesh_fn)
    mesh.GetPointData().RemoveArray('Normals_')
    com_filter = vtk.vtkCenterOfMass()
    com_filter.SetInputData(mesh)
    com_filter.Update()
    center = com_filter.GetCenter()

    poly_la = thresholdPolyData(mesh, threshold_name, (2, 2),'point')
    poly_lv = thresholdPolyData(mesh, threshold_name, (0, 0),'point')
    poly_rv = thresholdPolyData(mesh, threshold_name, (1, 1),'point')

    view_up = np.mean(vtk_to_numpy(poly_la.GetPoints().GetData()), axis=0) - \
            np.mean(vtk_to_numpy(poly_lv.GetPoints().GetData()), axis=0)
    view_horizontal = np.mean(vtk_to_numpy(poly_rv.GetPoints().GetData()), axis=0) - \
            np.mean(vtk_to_numpy(poly_lv.GetPoints().GetData()), axis=0)
    view_up /= np.linalg.norm(view_up)
    nrm = np.cross(view_up, view_horizontal)
    nrm /= np.linalg.norm(nrm)
    
    # tilt the view a little bit
    view_up = view_up + tilt_factor * nrm
    view_up /= np.linalg.norm(view_up)
    nrm = np.cross(view_up, view_horizontal)
    nrm /= np.linalg.norm(nrm)

    # compute extend
    extend = np.matmul(vtk_to_numpy(mesh.GetPoints().GetData()) - np.expand_dims(np.array(center), 0), view_up.reshape(3,1))
    extend_size = np.max(extend) - np.min(extend)
    nrm *= np.linalg.norm(view_up)*extend_size*2.3

    mesh_mapper = vtk.vtkPolyDataMapper()
    if id_to_visualize > -1:
        poly_myo = thresholdPolyData(mesh, threshold_name, (id_to_visualize, id_to_visualize), 'point')
        mesh_mapper.SetInputData(poly_myo)
    else:
        mesh_mapper.SetInputData(mesh)
    mesh_mapper.SelectColorArray(array_name)
    mesh_mapper.SetScalarModeToUsePointFieldData()
    mesh_mapper.RemoveVertexAttributeMapping('Normals_')
    
    # need the following line for categorical color map
    lut = setupLuts('set3')
    rng = mesh.GetPointData().GetArray(array_name).GetRange()
    lut.SetTableRange(rng)
    labels = np.unique(vtk_to_numpy(mesh.GetPointData().GetArray(array_name)))
    lut.SetNumberOfTableValues(len(labels))
    values = vtk.vtkVariantArray()
    for i in range(len(labels)):
        v = vtk.vtkVariant(int(labels[i]))
        values.InsertNextValue(v)
    for i in range(values.GetNumberOfTuples()):
        lut.SetAnnotation(i, values.GetValue(i).ToString())

    mesh_mapper.SetLookupTable(lut)
    mesh_mapper.SetScalarRange(mesh.GetPointData().GetArray(array_name).GetRange())
    actor = vtk.vtkActor()
    actor.SetMapper(mesh_mapper)

    prop = vtk.vtkProperty()
    prop.SetEdgeVisibility(show_edge)
    prop.SetOpacity(1)
    prop.SetAmbient(0.2)
    #prop.SetRenderLinesAsTubes(True)
    actor.SetProperty(prop)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    colors = vtk.vtkNamedColors()
    ren.SetBackground(colors.GetColor3d("White"))
    if camera is None:
        camera = ren.MakeCamera()
        camera.SetClippingRange(0.1, 1000)
        camera.SetFocalPoint(*center)
        camera.SetViewUp(*view_up)
        camera.SetPosition(center[0]+nrm[0], center[1]+nrm[1], center[2]+nrm[2])
    ren.SetActiveCamera(camera)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize((digit_size, digit_size))
    # use this line to generate figure
    ren_win.Render()
    # use the following lines to start an iterative window
    #iren = vtk.vtkRenderWindowInteractor()
    #iren.SetRenderWindow(ren_win)
    #iren.Start() 
    return ren_win, camera

def write_image(ren_win, filename):
    print("Writing PNG to : ", filename)
    w2im = vtk.vtkWindowToImageFilter()
    w2im.SetInput(ren_win)
    w2im.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(w2im.GetOutputPort())
    writer.Write()

def generate_figure(mesh_fn, digit_size, id_to_visualize=-1, tilt_factor=0.5, camera=None):
    ren_win, camera = visualize_template(mesh_fn, array_name='RegionId', threshold_name='RegionId', show_edge=False, id_to_visualize=id_to_visualize, tilt_factor=tilt_factor, digit_size=digit_size, camera=camera)
    # convert to image
    w2im = vtk.vtkWindowToImageFilter()
    w2im.SetInput(ren_win)
    w2im.Update()
    im = w2im.GetOutput()
    img_scalar = im.GetPointData().GetScalars()
    dims = im.GetDimensions()
    n_comp = img_scalar.GetNumberOfComponents()
    temp = vtk_to_numpy(img_scalar)
    numpy_data = temp.reshape(dims[1],dims[0],n_comp)
    numpy_data = numpy_data.transpose(0,1,2)
    numpy_data = np.flipud(numpy_data)
    return numpy_data.astype(int), camera

def plot_results_for_a_sample(dir_n, gt_dir_n, sample_n, camera_t=None, camera_t_myo=None):
    size = 512
    row = 3
    col = 3
    figure = np.ones((size * row, size * col, 3)).astype(int)*255
    
    ground_truth_fn = os.path.join(gt_dir_n, sample_n + '.vtp')
    final_prediction_fn = os.path.join(dir_n, sample_n + '_pred.vtp')
    prediction_noDs_fn_list = sorted(glob.glob(os.path.join(dir_n, sample_n + '_pred_noCorr*.vtp')))
    type_prediction_fn = os.path.join(dir_n, sample_n + '_pred_type.vtp')

    # first row (wh): gt; final pred; type
    pred_data, camera1 = generate_figure(final_prediction_fn, size, tilt_factor=0.5)
    gt_data, _ = generate_figure(ground_truth_fn, size, camera=camera1, tilt_factor=0.5)
    type_data, camera_t = generate_figure(type_prediction_fn, size, camera=camera_t, tilt_factor=0.5)
    figure[:size, :size, :] = gt_data 
    figure[:size, size:2*size, :] = pred_data
    figure[:size, 2*size:3*size, :] = type_data
    plt.text(0+size//3, 0, 'Ground Truth')
    plt.text(size+size//3, 0, 'Prediction')
    plt.text(2*size+size//3, 0, 'Type')
    
    # second row (Myo): gt; final_pred, type
    pred_data, camera2 = generate_figure(final_prediction_fn, size, tilt_factor=-1., id_to_visualize=4)
    gt_data, _ = generate_figure(ground_truth_fn, size, camera=camera2, tilt_factor=-1., id_to_visualize=4)
    type_data, camera_t_myo = generate_figure(type_prediction_fn, size, camera=camera_t_myo, tilt_factor=-1., id_to_visualize=4)
    figure[size:2*size,:size, :] = gt_data
    figure[size:2*size, size:2*size, :] = pred_data
    figure[size:2*size, 2*size:3*size, :] = type_data
    plt.text(0+size//3, size+size//5, 'Ground Truth')
    plt.text(size+size//3, size+size//5, 'Prediction')
    plt.text(2*size+size//3, size+size//5, 'Type')
    
    # thrid row (wh): prednoDs
    for i, fn in enumerate(prediction_noDs_fn_list):
        pred_data, _ = generate_figure(fn, size, tilt_factor=0.5, camera=camera1)
        figure[2*size:3*size, i*size:(i+1)*size, :] = pred_data
        plt.text(i*size+size//3,2*size, 'Pred - dS B{}'.format(i))

    plt.imshow(figure)
    plt.axis('off')
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 18.5)
    fig.savefig(os.path.join(dir_n, sample_n+'.png'), dpi=300)
    return camera_t, camera_t_myo

if __name__ == '__main__':
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_AlterTypeLaterNrmWt0.02_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_AlterTypeLaterNrmWt0.005_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.1_FreezeT20Every5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.1_FreezeT20Every5Decay_smpleFac5'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1blockTypeInd_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1blockSDVInt_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned',
            '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned_GEN',
            '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned_GEN_med3']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_preTrainT_f32']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_preTrainT_f32', 
            '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_preTrainT_f32']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_SegWt50Clamp0.5_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_SegWt50Clamp0.5_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_preTrainT_f32', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_preTrainT_f32']
    fn_list = ['dice.csv', 'dice.csv', 'dice_unet.csv', 'dice.csv', 'dice_unet.csv']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned_SDF', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned_SDF_BinaryRegr']
    fn_list = ['dice.csv', 'dice.csv', 'dice.csv']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5']
    fn_list = ['dice.csv']
    '''
    # comparison between before and after changing to binary rather than sdf
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned_SDF_BinaryRegr', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_SegWt50Clamp0.5_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_SDF_sprvsd_SegWt50Clamp0.5_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2']
    fn_list = ['dice.csv', 'dice.csv', 'dice_unet.csv', 'dice.csv', 'dice_unet.csv']
    # comparison between before and after adding normal regularization between type and deformed
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLossWt0.01', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLossWt0.01', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLossWt0.001', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLossWt0.001']
    fn_list = ['dice.csv', 'dice_unet.csv', 'dice.csv', 'dice_unet.csv', 'dice.csv', 'dice_unet.csv']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLoss0.01E20_smplInteriorFix', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLoss0.01E20_smplInteriorFix', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLoss0.01E20_smplBoundary', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLoss0.01E20_smplBoundary', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLoss0.01E20_smplMarginType2', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/Seg/Unet_Flow_Binary_sprvsd_SegWt50_PosEnc2_nrmLoss0.01E20_smplMarginType2']
    fn_list = ['dice.csv', 'dice_unet.csv', 'dice.csv', 'dice_unet.csv', 'dice.csv', 'dice_unet.csv']
    #dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_aligned_noAlter_smpleFac20']
    #dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_aligned_noAlter_smpleFac20', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_aligned_smpleFac20']
    '''
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_initx0.1L10_x', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_initx0.1L10_x']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter10', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter10', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter5', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter5']
    fn_list = ['dice_noCorr.csv', 'dice_unet.csv', 'dice_noCorr.csv', 'dice_unet.csv']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter5FullySep', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter10FullySep', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter15FullySep', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs_nrmNodeType_typefreeze_alter10FullySep_diff10']
    fn_list = ['dice_noCorr.csv', 'dice_noCorr.csv', 'dice_noCorr.csv', 'dice_noCorr.csv']
    #fn_list = ['dice.csv', 'dice.csv']
    fig, ax = plt.subplots()
    #
    #for dir_n, fn in zip(dir_n_list, fn_list):
    #    print(dir_n, fn)
    #    plot_dice_scores_over_epochs(dir_n, ax, fn)
    #plt.show()
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_aligned_noAlter_smpleFac20/aligned_test_70', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/MUnetBase16_aligned/aligned_test_100']
    method_list = ['SDF', 'UNet']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_BCEwLogits/validate_90/dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5Ds0.1_FixedAct/validate_70/dice_unet.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5Ds0.1_FixedAct/validate_70/dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5Ds0.1_FixedAct/validate_70/dice_noCorr.csv']
    method_list = ['UNet', 'Intermediate UNet', 'After correction', 'After deformation']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_BCEwLogits/validate_40/dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs/validate_40/dice_unet.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeMyo/Unet_Flow_BCE_sprvsd_PosEnc2_smplBound50_Typelrx2_param5_FixedAct_noDs/validate_40/dice_noCorr.csv']
    method_list = ['UNet', 'Intermediate UNet', 'After deformation']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/UNet_BCE_noclosing/test/exclude_65/dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Final/Unet_Flow_BCE_sprvsd_PosEnc2_alter5FullySep_fixT95trainNodeSeg/test/exclude_115/dice_unet.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Final/Unet_Flow_BCE_sprvsd_PosEnc2_alter5FullySep_fixT95trainNodeSeg/test/exclude_115/dice_noCorr.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Final/Unet_Flow_BCE_sprvsd_PosEnc2_alter5FullySep_fixT95trainNodeSeg_testOpt/test/exclude_109_trained_on_all_samples_with_aug/dice_noCorr.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Final/Unet_Flow_BCE_sprvsd_PosEnc2_alter5FullySep_fixT95trainNodeSeg_testOpt/test/exclude_109_trained_per_sample_iter200/dice_noCorr.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeJune20MyoAug1/Final/Unet_Flow_BCE_sprvsd_PosEnc2_alter5FullySep_fixT95trainNodeSeg_testOpt/test/exclude_115_trained_per_sample_iter1000/dice_noCorr.csv']
    method_list = ['UNet', 'Intermediate UNet', 'After deformation', 'Test-time opt', 'Test-time opt per sample 200iters', 'Test-time opt per sample 1000iters']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_opt_2000/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/NDF_alter5Joint1000_probSmpl_PosBdRg2_latent4_Layer6_FlowMag0_noWtDcy/testopt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/DeepSDF_TypeSpatial/testopt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/DeepSDF_TypeSpatial_latent4_noCond/testopt_2000/dice_test.csv']
    method_list = ['Ours', 'Ours no correction', 'NDF', 'Conditional DeepSDF', 'DeepSDF']
    #dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_fixType_opt_2000/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_fixType_opt_2000/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_opt_2000/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_nodeExpFac1_healthyInit_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_nodeExpFac1_healthyInit_opt_2000/dice_noCorr_test.csv']
    #method_list = ['Lat8+corr', 'Lat8', 'Lat4+corr', 'Lat4', 'Lat4+optT+corr', 'Lat4+optT', 'Lat4+optHealthyT+corr', 'Lat4+optHealthyT']
    #dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_fixType_opt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_bothExpFac1_fixType_opt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/NDF_alter5Joint1000_probSmpl_PosBdRg2_latent4_Layer6_FlowMag0_noWtDcy/testopt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/DeepSDF_TypeSpatial/testopt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/DeepSDF_TypeSpatial_latent4_noCond/testopt_2000/vsd/vsd_dice.csv']
    #method_list = ['Ours-Lat8', 'Ours', 'Ours local', 'NDF', 'CDeepSDF', 'DeepSDF']
    #dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/vsd_noCorr/vsd_noCorr_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/NDF_alter5Joint1000_probSmpl_PosBdRg2_latent4_Layer6_FlowMag0_noWtDcy/test_lat8_opt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/DeepSDF_TypeSpatial/test_lat8_opt_2000/vsd/vsd_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/DeepSDF_TypeSpatial_latent4_noCond/test_lat8_opt_2000/vsd/vsd_dice.csv']
    #method_list = ['Ours+Corr', 'Ours', 'NDF', 'CDeepSDF', 'DeepSDF']
    #dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent2_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent8_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/dice_test.csv']
    #method_list = ['dim=2', 'dim=4', 'dim=8']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat8_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy/test_both_lat4_fixType_opt_2000/dice_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final/SeparateDs/alter5Joint1000_probSmpl_typeSpatialFeatAugMul_PosBdRg2_latent4_Layer6Type6_DStypeDep_FlowMag0_noWtDcy_2ShapeCodeRe/test_both_lat8_fixType_opt_2000/dice_test.csv']
    method_list = ['dim=2', 'dim=4', 'dim=8']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeWHSep3/Register_Unet_Flow/test_both_lat4_fixType_opt_2000/Compared_with_gt_dice.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeWHSep3/Register_Unet_Flow/test_both_lat4_fixType_opt_2000/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/SegLargeWHSep3/Unet/test_80/dice.csv']
    method_list = ['Registered', 'UNet Fitting', 'UNet']
    method_list = ['Full', '15 Slices', '10 Slices', '5 Slices', '3 Slices']
    dir_n_list = ['/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/test_both_lat4_fixType_opt_2500/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/test_both_lat4_fixType_opt_2500_sparse15/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/test_both_lat4_fixType_opt_2500_sparse10/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/test_both_lat4_fixType_opt_2500_sparse5/dice_noCorr_test.csv', '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/test_both_lat4_fixType_opt_2500_sparse3/dice_noCorr_test.csv']
    plot_dice_scores_as_boxplot(dir_n_list, method_list, ax)
    plt.show()
    #sys.exit()
    #mesh_fn = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_smpleFac20/test_10/ct_1145_image_pred.vtp'
    #visualize_template(mesh_fn, array_name='RegionId', show_edge=False, id_to_visualize=4, tilt_factor=-0.5)
    #generate_figure(mesh_fn, digit_size=512)

    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/3blocks_bgFix_DSB3B2B1WT2n8n5n5_smpleFac20/small_test_20'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_n') 
    args = parser.parse_args()
    gt_dir_n = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all/imageCHDcleaned_all/vtk'
    gt_dir_n = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all_aligned/imageCHDcleaned_all/vtk'
    #gt_dir_n = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all_aligned_rigid/vtk'
    sample_n_list = ['ct_1102_image', 'ct_1103_image', 'ct_1145_image', 'ct_1146_image']
    sample_n_list = ['ct_1024_image', 'ct_1052_image', 'ct_1060_image', 'ct_1067_image', \
            'ct_1077_image', 'ct_1081_image', 'ct_1101_image', 'ct_1102_image', \
            'ct_1103_image', 'ct_1145_image', 'ct_1146_image', 'ct_1147_image']
    fns = glob.glob(os.path.join(args.dir_n, '*_pred.vtp'))
    sample_n_list = [os.path.basename(f[:-9]) for f in fns]
    print(fns)
    print(sample_n_list)
    for sample_n in sample_n_list:
        try:
            plot_results_for_a_sample(args.dir_n, gt_dir_n, sample_n)
        except Exception as e:
            print(e)
