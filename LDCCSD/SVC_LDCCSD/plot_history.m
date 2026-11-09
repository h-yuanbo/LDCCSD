function plot_history(history, version)
    figure; plot(history.obj, 'LineWidth',1.5);
    xlabel('Iteration'); ylabel('Objective'); title(['Objective trajectory ', version]);
    
    figure; plot(history.u_stat, 'LineWidth',1.5);
    xlabel('Iteration'); ylabel('u values');
    legend('min(u)','median(u)','max(u)');
    title(['Evolution of u distribution ', version]);
end