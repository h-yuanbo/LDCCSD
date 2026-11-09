function plot_historywithoutu(history, version)
    if any(history.obj ~= 0)
        figure; plot(history.obj, 'LineWidth',1.5);
        xlabel('Iteration'); ylabel('Objective'); 
        title(['Objective trajectory ', version]);
    end
end