for file in $(find results_figures -name \*.png) ; do
    echo "Copying $file to ../thesis_doc/figures/"
    cp $file ../thesis_doc/figures/

done
