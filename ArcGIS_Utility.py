__author__ = 'Z0ki'
import arcpy
import ContourTree
import os

def GetContourID_Contour_Area(infc,areaThresholdValue):
    arcpy.AddField_management(infc, "Cont_ID", "LONG")
    arcpy.AddField_management(infc, "Cont_", "DOUBLE")
    arcpy.AddField_management(infc, "Lap_Area", "DOUBLE")

    feature_array = arcpy.Array()
    count = 0

    rows = arcpy.UpdateCursor(infc)
    for row in rows:

        # Create the geometry object
        feature = row.getValue("Shape")
        firstPnt = feature.firstPoint
        lastPnt = feature.lastPoint

        if  (firstPnt.X == lastPnt.X) & (firstPnt.Y == lastPnt.Y):
            count += 1
            #arcpy.AddMessage count
            feature_array.removeAll()
            for part in feature:
                for pnt in part:
                    if pnt:
                        feature_array.add(pnt)
                    else:
                    # If pnt is None, this represents an interior ring
                        arcpy.AddMessage("Interior Ring:")
            polygon = arcpy.Polygon(feature_array)
            area = polygon.area
            if(area < areaThresholdValue):
                area = 0
        else:
           area = 0
        row.setValue("Lap_Area",area)
        row.setValue("Cont_ID",row.getValue("FID"))
        row.setValue("Cont_",row.getValue("Contour"))
        rows.updateRow(row)
    outTempLayer = os.path.join(os.path.split(infc)[0],"tempLayer")
    arcpy.MakeFeatureLayer_management(infc,outTempLayer)
    selectExpression = arcpy.AddFieldDelimiters(outTempLayer,"Lap_Area") + "=0"
    arcpy.SelectLayerByAttribute_management(outTempLayer,"NEW_SELECTION",selectExpression)
    arcpy.DeleteFeatures_management(outTempLayer)

    arcpy.AddMessage("GetContourID_Contour_Area Over!")

def CalculateContour_Line_FromLine(outContours,gMinLength):
    arcpy.CheckOutExtension("Spatial")
    arcpy.AddField_management(outContours,"length","DOUBLE","#","#","#","#","NULLABLE","NON_REQUIRED","#")
    arcpy.CalculateField_management(outContours,"length","!shape.length@meters!","PYTHON_9.3","#")

    arcpy.AddField_management(outContours, "Distance", "DOUBLE")
    expression = "MySub( !Shape!)"
    codeblock = """def MySub(shape):
        fst = arcpy.PointGeometry(shape.firstPoint)
        lst = arcpy.PointGeometry(shape.lastPoint)
        return float(fst.distanceTo(lst))"""
    arcpy.CalculateField_management(outContours, "Distance", expression, "PYTHON_9.3", codeblock)


    tempLayer = os.path.join(os.path.split(outContours)[0],"temp_Line_Layer")
    arcpy.MakeFeatureLayer_management(outContours, tempLayer)
    expression = "length < " + str(gMinLength) + " or Distance > 0.01"
    # Execute SelectLayerByAttribute to determine which features to delete
    arcpy.SelectLayerByAttribute_management(tempLayer, "NEW_SELECTION",expression)

    # Execute GetCount and if some features have been selected, then
    #  execute DeleteFeatures to remove the selected features.
    if int(arcpy.GetCount_management(tempLayer).getOutput(0)) > 0:
        arcpy.DeleteFeatures_management(tempLayer)
    arcpy.Delete_management(tempLayer)
    arcpy.AddMessage("CalculateContour_Line Over!")
    #arcpy.AddMessage("CalculateContour_Line Over!")





def CalculateContour_Line(inRaster, outContours, contourInterval, baseContour,gMinLength):
    arcpy.CheckOutExtension("Spatial")
    tempLayer = os.path.join(os.path.split(outContours)[0],"temp_Line_Layer.shp")
    maxValue = int(arcpy.Raster(inRaster).maximum) + 1
    contour_list = []
    for i in range(0,int(int(maxValue-baseContour)/contourInterval) + 1):
        c_value = baseContour + i * contourInterval
        contour_list.append(c_value)


    arcpy.sa.ContourList(inRaster,tempLayer,contour_list)
    #arcpy.sa.Contour(inRaster, tempLayer, contourInterval, baseContour)
    #arcpy.sa.ContourWithBarriers(inRaster,outContours,"","POLYLINES","","",baseContour,contourInterval)
    arcpy.AddField_management(tempLayer,"length","DOUBLE","#","#","#","#","NULLABLE","NON_REQUIRED","#")
    arcpy.CalculateField_management(tempLayer,"length","!shape.length@meters!","PYTHON_9.3","#")

    arcpy.AddField_management(tempLayer, "Distance", "DOUBLE")
    expression = "MySub(!Shape!)"
    codeblock = """def MySub(shape):
        fst = arcpy.PointGeometry(shape.firstPoint)
        lst = arcpy.PointGeometry(shape.lastPoint)
        return float(fst.distanceTo(lst))"""
    arcpy.CalculateField_management(tempLayer, "Distance", expression, "PYTHON_9.3", codeblock)


    tempLayer2 = os.path.join(os.path.split(outContours)[0],"temp_Line_Layer2")
    arcpy.MakeFeatureLayer_management(tempLayer, tempLayer2)
    expression = "length > " + str(gMinLength) + " and Distance < 0.01"
    # Execute SelectLayerByAttribute to determine which features to delete
    arcpy.SelectLayerByAttribute_management(tempLayer2, "NEW_SELECTION",expression)

    # Execute GetCount and if some features have been selected, then
    #  execute DeleteFeatures to remove the selected features.
    if int(arcpy.GetCount_management(tempLayer2).getOutput(0)) > 0:
        arcpy.CopyFeatures_management(tempLayer2,outContours)
    arcpy.Delete_management(tempLayer)
    arcpy.Delete_management(tempLayer2)
    arcpy.AddMessage("CalculateContour_Line Over!")
    #arcpy.AddMessage("CalculateContour_Line Over!")

def GetContourID_Contour_Area2(infc,outfc,areaThresholdValue):
    workspace = os.path.split(outfc)[0]
    outfcName = os.path.split(outfc)[1]

    feature_array = arcpy.Array()
    #count = 0
    origin_des = arcpy.Describe(infc)
    sr = origin_des.spatialReference
    arcpy.CreateFeatureclass_management(workspace,outfcName,"POLYGON","","","",sr)

    arcpy.AddField_management(infc, "Cont_ID", "LONG")
    arcpy.AddField_management(infc, "Cont_", "DOUBLE")
    arcpy.AddField_management(infc, "Lap_Area", "DOUBLE")

    arcpy.AddField_management(outfc, "Cont_ID", "LONG")
    arcpy.AddField_management(outfc, "Cont_", "DOUBLE")
    arcpy.AddField_management(outfc, "Lap_Area", "DOUBLE")
    # arcpy.AddField_management(outfc,"Lap_Area","DOUBLE","#","#","#","#","NULLABLE","NON_REQUIRED","#")

    cursor = arcpy.InsertCursor(outfc)
    rows = arcpy.UpdateCursor(infc)
    for row in rows:
        feature = row.getValue("Shape")
        Cont_ID = int(row.getValue("ID")) + 1
        feature_array.removeAll()
        for part in feature:
            for pnt in part:
                if pnt:
                    feature_array.add(pnt)

        f_polygon = arcpy.Polygon(feature_array,sr)

        area = f_polygon.area
        if (area >= float(areaThresholdValue)):
            newrow = cursor.newRow()
            newrow.SHAPE = f_polygon
            newrow.setValue("Cont_ID",Cont_ID)
            newrow.setValue("Cont_",row.getValue("Contour"))
            newrow.setValue("Lap_Area",area)
            cursor.insertRow(newrow)
        else:
            area = 0

        row.setValue("Cont_",row.getValue("Contour"))
        row.setValue("Lap_Area",area)
        row.setValue("Cont_ID",Cont_ID)
        rows.updateRow(row)
    del cursor

    outTempLayer = os.path.join(os.path.split(infc)[0],"tempLayer")
    arcpy.MakeFeatureLayer_management(infc,outTempLayer)
    selectExpression = "Lap_Area = 0 "
    arcpy.SelectLayerByAttribute_management(outTempLayer,"NEW_SELECTION",selectExpression)
    if int(arcpy.GetCount_management(outTempLayer).getOutput(0)) > 0:
        arcpy.DeleteFeatures_management(outTempLayer)
    arcpy.Delete_management(outTempLayer)
    arcpy.AddMessage("GetContourID_Contour_Area Over!")

def CalculateContour_Polygon(inContour_Line,outContour_Polygon,clusTol):
    arcpy.FeatureToPolygon_management(inContour_Line, outContour_Polygon,clusTol,"ATTRIBUTES", "")

    arcpy.AddMessage("CalculateContour_Polygon Over!")

def GetContourID4Polygon(inContourPolygon,inContourLapPolygon, outContourPolygonWithID,ToLevelID):
    if (os.path.splitext(inContourPolygon)[1].lower() == ".shp"):
        outContours_Polygon_WithID_Temp = os.path.join(os.path.split(outContourPolygonWithID)[0] ,"IDpolygonTemp.shp")
    else:
        outContours_Polygon_WithID_Temp = os.path.join(os.path.split(outContourPolygonWithID)[0] ,"IDpolygonTemp")
    fieldmappings = arcpy.FieldMappings()
    fieldmappings.addTable(inContourLapPolygon)
    fieldmappings.addTable(inContourPolygon)
    arcpy.SpatialJoin_analysis(inContourPolygon,inContourLapPolygon,outContours_Polygon_WithID_Temp,"JOIN_ONE_TO_ONE","KEEP_ALL",fieldmappings,"WITHIN_CLEMENTINI")
    arcpy.Dissolve_management(outContours_Polygon_WithID_Temp,outContourPolygonWithID,["Cont_ID"],[["Cont_","FIRST"],["Lap_Area","SUM"]])
    arcpy.Delete_management(outContours_Polygon_WithID_Temp)

    arcpy.AddField_management(outContourPolygonWithID, "Line_Cou", "LONG")
    arcpy.AddField_management(outContourPolygonWithID,"Level_ID","SHORT")
    arcpy.AddField_management(outContourPolygonWithID,"Level_BD","SHORT")
    arcpy.AddField_management(outContourPolygonWithID,"LabelFlag","SHORT")
    arcpy.AddField_management(outContourPolygonWithID,"Out_Nbr_Ct","SHORT")
    arcpy.AddField_management(outContourPolygonWithID,"Out_Nbr_ID","SHORT")
    arcpy.AddField_management(outContourPolygonWithID,"Contain_ID","TEXT")

    expression = "MySub( !Shape!)"
    codeblock = """def MySub(feat):
        if feat.isMultipart:
            return 2
        else:
            return 1"""
    arcpy.CalculateField_management(outContourPolygonWithID, "Line_Cou", expression, "PYTHON_9.3", codeblock)
    arcpy.AddMessage("GetContourID4Polygon Over!")

def AlterField_management(inFeatureClass, outFeatureClass, new_name_by_old_name):
    """ Renames specified fields in input feature class/table
    :table:                 input table (fc, table, layer, etc)
    :out_table:             output table (fc, table, layer, etc)
    :new_name_by_old_name:  {'old_field_name':'new_field_name',...}
    ->  out_table
    """
    existing_field_names = [field.name for field in arcpy.ListFields(inFeatureClass)]

    field_mappings = arcpy.FieldMappings()
    field_mappings.addTable(inFeatureClass)

    for old_field_name, new_field_name in new_name_by_old_name.iteritems():
        if old_field_name not in existing_field_names:
            message = "Field: {0} not in {1}".format(old_field_name, inFeatureClass)
            raise Exception(message)

        mapping_index = field_mappings.findFieldMapIndex(old_field_name)
        field_map = field_mappings.fieldMappings[mapping_index]
        output_field = field_map.outputField
        output_field.name = new_field_name
        output_field.aliasName = new_field_name
        field_map.outputField = output_field
        field_mappings.replaceFieldMap(mapping_index, field_map)

    # use merge with single input just to use new field_mappings
    arcpy.Merge_management(inFeatureClass, outFeatureClass, field_mappings)
    return outFeatureClass

def GetPolygonNeighbor(inContour_Polygon_ID, outContours_Polygon_Neigh,outContours_Polygon_Summarize):
    workSpace = os.path.split(outContours_Polygon_Neigh)[0]
    if (os.path.splitext(outContours_Polygon_Neigh)[1].lower() == ".dbf"):
        outContours_Polygon_Neigh_Temp = os.path.join(workSpace,"tempNei.dbf")
        arcpy.PolygonNeighbors_analysis(inContour_Polygon_ID,outContours_Polygon_Neigh_Temp,"FID;Cont_ID;FIRST_Cont;SUM_Lap_Ar", "NO_AREA_OVERLAP", "BOTH_SIDES")
        arcpy.DeleteField_management(outContours_Polygon_Neigh_Temp,["src_FID","nbr_FID","LENGTH","NODE_COUNT"])
        new_name_by_old_name = { 'src_FIRST_':'src_Cont','nbr_FIRST_':'nbr_Cont','src_SUM_La':'src_Area','nbr_SUM_La':'nbr_Area' }
    else:
        outContours_Polygon_Neigh_Temp = os.path.join(workSpace,"tempNei")
        arcpy.PolygonNeighbors_analysis(inContour_Polygon_ID,outContours_Polygon_Neigh_Temp,"Cont_ID;FIRST_Cont_;SUM_Lap_Area", "NO_AREA_OVERLAP", "BOTH_SIDES")
        arcpy.DeleteField_management(outContours_Polygon_Neigh_Temp,["LENGTH","NODE_COUNT"])
        new_name_by_old_name = { 'src_FIRST_Cont_':'src_Cont','nbr_FIRST_Cont_':'nbr_Cont','src_SUM_Lap_Area':'src_Area','nbr_SUM_Lap_Area':'nbr_Area' }

    AlterField_management(outContours_Polygon_Neigh_Temp,outContours_Polygon_Neigh,new_name_by_old_name)
    arcpy.Delete_management(outContours_Polygon_Neigh_Temp)

    workSpace2 = os.path.split(outContours_Polygon_Summarize)[0]
    if (os.path.splitext(outContours_Polygon_Summarize)[1].lower() == ".dbf"):
        outContours_Polygon_Summarize_Temp = os.path.join(workSpace2,"tempSum.dbf")
        arcpy.Statistics_analysis(outContours_Polygon_Neigh,outContours_Polygon_Summarize_Temp,[["src_Cont_I", "COUNT"]],"src_Cont_I")
        arcpy.DeleteField_management(outContours_Polygon_Summarize_Temp,"COUNT_src_")
    else:
        outContours_Polygon_Summarize_Temp = os.path.join(workSpace2,"tempSum")
        arcpy.Statistics_analysis(outContours_Polygon_Neigh,outContours_Polygon_Summarize_Temp,[["src_Cont_ID", "COUNT"]],"src_Cont_ID")
        arcpy.DeleteField_management(outContours_Polygon_Summarize_Temp,"COUNT_src_Cont_ID")

    AlterField_management(outContours_Polygon_Summarize_Temp,outContours_Polygon_Summarize,{'FREQUENCY':'NBR_CNT'})
    arcpy.Delete_management(outContours_Polygon_Summarize_Temp)
    arcpy.AddMessage("GetPolygonNeighbor Over!")

def GetStatisticalDataTable(inDEMRaster,inContoutPolygonID,outStatisticalTable):
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.cellSize = 30
    arcpy.sa.ZonalStatisticsAsTable(inContoutPolygonID,"Cont_ID",inDEMRaster,outStatisticalTable,"DATA","ALL")
    arcpy.env.cellSize = inDEMRaster
    arcpy.AddMessage("GetStatisticalDataTable Over!")


def numpyToArray(AllInfo,inContour_Poly_ID):
    if (os.path.splitext(inContour_Poly_ID)[1].lower() == ".shp"):
        outTempTable = os.path.join(os.path.split(inContour_Poly_ID)[0],"tempTable.dbf")
    else:
        outTempTable = os.path.join(os.path.split(inContour_Poly_ID)[0],"tempTable")
    arcpy.da.NumPyArrayToTable(AllInfo, outTempTable)
    arcpy.JoinField_management(inContour_Poly_ID,"Cont_ID",outTempTable,"Cont_ID")
    arcpy.CalculateField_management(inContour_Poly_ID,"Level_ID","!Level_ID_1!","PYTHON_9.3")
    arcpy.CalculateField_management(inContour_Poly_ID,"Level_BD","!Level_BD_1!","PYTHON_9.3")
    #arcpy.RemoveJoin_management(inContour_Poly_ID, outTempTable)
    arcpy.Delete_management(outTempTable)


def LapJoin(inContour_Polygon_ID,inContour_Polygon_Lap_temp):
    # arcpy.Dissolve_management(inContour_Polygon_Lap,inContour_Polygon_Lap_temp,["Cont_ID"],[["Cont_","FIRST"],["Lap_Area","SUM"]])
    # arcpy.Delete_management(inContour_Polygon_Lap)
    arcpy.AddIndex_management (inContour_Polygon_Lap_temp, "Cont_ID", "Cont_Index")
    arcpy.JoinField_management(inContour_Polygon_Lap_temp,"Cont_ID",inContour_Polygon_ID,"Cont_ID",["Level_BD"])


def SingleShapefile(inContour_Polygon_Lap_temp,OutPath,Max_level,inDEMRaster,isContain):
    if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
        arcpy.CreateFolder_management(OutPath, "Single")
    else:
        sr = arcpy.Describe(inContour_Polygon_Lap_temp).spatialReference
        arcpy.CreateFeatureDataset_management(OutPath, "Single",sr)

    Single_OutPath = os.path.join(OutPath,"Single")
    for level_id in range(1,int(Max_level)):
        selection = "Level_BD = " + str(level_id)
        if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
            shapefileName = "Single_L" + str(level_id) + ".shp"
            outputTemp = os.path.join(Single_OutPath,"temp_" + str(level_id) + ".shp")
        else:
            shapefileName = "Single_L" + str(level_id)
            outputTemp = os.path.join(Single_OutPath,"temp_" + str(level_id))
        outputShapefile = os.path.join(Single_OutPath,shapefileName)

        arcpy.MakeFeatureLayer_management (inContour_Polygon_Lap_temp,outputTemp )
        arcpy.SelectLayerByAttribute_management (outputTemp, "NEW_SELECTION", selection)
        arcpy.CopyFeatures_management(outputTemp,outputShapefile)
        arcpy.Delete_management(outputTemp)

    for level_id in range(1,int(Max_level)):
        if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
            shapefileName = "Single_L" + str(level_id) + ".shp"
        else:
            shapefileName = "Single_L" + str(level_id)
        outputShapefile = os.path.join(Single_OutPath,shapefileName)
        temp_level_id = level_id
        while(temp_level_id < int(Max_level)-1):
            temp_level_id += 1
            if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
                shapefileNameTemp = "Single_L" + str(temp_level_id) + ".shp"
                out_feature_class  = os.path.join(Single_OutPath,"SingleSJTemp.shp")
            else:
                shapefileNameTemp = "Single_L" + str(temp_level_id)
                out_feature_class  = os.path.join(Single_OutPath,"SingleSJTemp")

            outputShapefileTemp = os.path.join(Single_OutPath,shapefileNameTemp)


            arcpy.SpatialJoin_analysis(outputShapefileTemp,outputShapefile,out_feature_class, "JOIN_ONE_TO_MANY","KEEP_COMMON",match_option="WITHIN")
            result = arcpy.GetCount_management(out_feature_class)
            overlayCount = int(result.getOutput(0))
            if overlayCount>0:
                if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
                    outEraseTemp = os.path.join(Single_OutPath,"EraseTemp.shp")
                else:
                    outEraseTemp = os.path.join(Single_OutPath,"EraseTemp")
                arcpy.Erase_analysis(outputShapefile,out_feature_class,outEraseTemp)
                arcpy.Delete_management(outputShapefile)
                arcpy.Rename_management(outEraseTemp,outputShapefile)
            arcpy.Delete_management(out_feature_class)


        arcpy.CalculateField_management(outputShapefile,"Lap_Area","!shape.area@squaremeters!","PYTHON_9.3","#")
        ContourTree.CalculateVolume(outputShapefile,inDEMRaster)

        if isContain:
            Contain_OutPath = os.path.join(OutPath,"Contains")
            if level_id == 1:
                if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
                    arcpy.CreateFolder_management(OutPath, "Contains")
                    contain_ShapefileName = "Contain_L" + str(level_id) + ".shp"
                else:
                    sr = arcpy.Describe(inContour_Polygon_Lap_temp).spatialReference
                    arcpy.CreateFeatureDataset_management(OutPath, "Contains",sr)
                    contain_ShapefileName = "Contain_L" + str(level_id)

                outputContainShapefile = os.path.join(Contain_OutPath,contain_ShapefileName)
                arcpy.CopyFeatures_management(outputShapefile,outputContainShapefile)
            else:
                if (os.path.splitext(inContour_Polygon_Lap_temp)[1].lower() == ".shp"):
                    contain_ShapefileName = "Contain_L" + str(level_id) + ".shp"
                    preShape = os.path.join(Contain_OutPath,"Contain_L" + str(level_id-1) + ".shp")
                else:
                    contain_ShapefileName = "Contain_L" + str(level_id)
                    preShape = os.path.join(Contain_OutPath,"Contain_L" + str(level_id-1))
                outputContainShapefile = os.path.join(Contain_OutPath,contain_ShapefileName)

                arcpy.Update_analysis(preShape,outputShapefile,outputContainShapefile)
                arcpy.CalculateField_management(outputContainShapefile,"Lap_Area","!shape.area@squaremeters!","PYTHON_9.3","#")
                ContourTree.CalculateVolume(outputContainShapefile,inDEMRaster)

def IndividualShapefile(inContour_Polygon_Lap_temp,OutPath,Max_level,isContain):
    flag = True
    for level_id in range(1,int(Max_level)):
        if(isContain):
            if flag:
                arcpy.CreateFolder_management(OutPath, "Contains")
                OutPath = os.path.join(OutPath,"Contains")
                flag = False
            selection = "Level_BD <= " + str(level_id) + " and Level_BD > 0"
            shapefileName = "Contains_L" + str(level_id) + ".shp"
        else:
            if flag:
                arcpy.CreateFolder_management(OutPath, "Single")
                OutPath = os.path.join(OutPath,"Single")
                flag = False
            selection = "Level_BD = " + str(level_id)
            shapefileName = "Single_L" + str(level_id) + ".shp"

        outputShapefile = os.path.join(OutPath,shapefileName)
        outputTemp = os.path.join(OutPath,"temp_" + str(level_id) + ".shp")
        arcpy.MakeFeatureLayer_management (inContour_Polygon_Lap_temp,outputTemp )
        arcpy.SelectLayerByAttribute_management (outputTemp, "NEW_SELECTION", selection)
        arcpy.CopyFeatures_management(outputTemp,outputShapefile)
        arcpy.Delete_management(outputTemp)


def CreateTIN(inLine,outTin):
    arcpy.CheckOutExtension("3D")
    arcpy.CreateTin_3d(outTin,in_features=inLine)

def mergeAllLevel(singleFolderPath, MaxLevel):
    list = []
    for i in reversed(range(1,MaxLevel+1)):
        singlePath = singleFolderPath + "\\Single_L"+str(i)+".shp"
        list.append(singlePath)
    OutputPath = singleFolderPath+ "\\AllLevel.shp"
    arcpy.Merge_management(list,OutputPath)