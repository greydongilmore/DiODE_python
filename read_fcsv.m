function [data_table] = read_fcsv(filename, subject)

    %--- Import the data
    fileID = fopen(fullfile(filename.folder, filename.name));
    data = textscan(fileID,'%s %f %f %f %f %f %f %f %f %f %f %f %s %s', 'Delimiter', ',','headerLines', 3, 'CollectOutput', 1);

    %--- Create table
    data_table = table;

    %--- Allocate imported array to column variable names
    data_table.fid = data{1,2}(:,11);
    data_table.X = data{1,2}(:,1);
    data_table.Y = data{1,2}(:,2);
    data_table.Z = data{1,2}(:,3);
    data_table.name = data{1,2}(:,11);
    data_table.description = data{1,3}(:,1);

end
