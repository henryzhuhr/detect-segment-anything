SAVE_DIR=.cache
if [ ! -d $SAVE_DIR ]; then
    mkdir -p $SAVE_DIR
fi

PROJECT_DIR=$(pwd)

cd $PROJECT_DIR/$SAVE_DIR
mkdir -p repo
cd repo

function clone_repo() {
    tdir=$1
    url=$2
    branch=$3
    if [ ! -d $tdir ]; then
        if [ ! -z $branch ]; then
            git clone -b $branch $url
        else
            git clone $url
        fi
    else
        cd $tdir
        # git pull
        cd ..
    fi
    if [ ! -d $tdir ]; then
        exit "Failed to clone $tdir"
    fi
}

# clone_repo "CLIP" "git@github.com:openai/CLIP.git"
# clone_repo "GroundingDINO" "git@github.com:IDEA-Research/GroundingDINO.git"
clone_repo "GroundingDINO" "git@github.com:HenryZhuHR/GroundingDINO.git" "dev"


cd $PROJECT_DIR/$SAVE_DIR
mkdir -p weights
cd weights
wget -nc -O groundingdino_swint_ogc.pth -c https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth