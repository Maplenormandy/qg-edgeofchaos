read -r -p "Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$  ]]
then
    echo "Clearing dirs"
    rm -rf ./snapshots/*
    rm -rf ./frames/*
    rm restart.h5
else
    echo "No cleaning"
fi


