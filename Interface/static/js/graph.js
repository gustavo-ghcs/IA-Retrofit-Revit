$(function () {
    let startPoint = [0,0];

    let generate = false;
    var roomSelect = 0;
    var mouseDown = false;
    var createLinew = false;

    $('#RightSVG').on('mousedown',function(e){
        console.log("Right!");
    })

    function clearHighLight(){
        var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke-width",0);
    }

    function circle_mousedown(){
        mouseDown = true;
        var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke-width",0);
        var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#"+this.id).attr("stroke-width",2);

        if(d3.event.button == 2){
            selectPoint.remove();
            mouseDown = false;

            var id = this.id.split("_")[1];

            var checkBtns = d3.select("body").selectAll(".checkButton");

            checkBtns.each(function(d,i){
                var btn_id = this.id.split("_")[1];
                if(id == btn_id){
                    var checkElement = this.parentElement.parentElement;
                    var todoList = document.getElementById('todo');
                    checkElement.classList.remove('checked');
                    deleteAnimation(checkElement);
                    todoList.insertBefore(checkElement, todoList.firstChild);
                }
            })
        }
    }

    function circle_mousemove(){
        if(mouseDown){
            let newX = d3.event.x - leftsvg.getBoundingClientRect().left;
            let newY = d3.event.y - leftsvg.getBoundingClientRect().top;
            var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#"+this.id)
                .attr("cx",newX/2).attr("cy",newY/2);
        }
    }

    function circle_mouseup(){
        mouseDown = false;
    }

    function deleteAnimation(deleteItem) {
        /* LEFT SIDE */
        const swirlR1 = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '0%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -100},
            radius: 30,
            swirlSize: 5,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: 90
        });
        const swirlR2 = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '0%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -85},
            radius: 25,
            swirlSize: 5,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: 70
        });
        const swirlR3 = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '0%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -70},
            radius: 20,
            swirlSize: 5,
            swirlFrequency: 1,
            duration: 1000,
            direction: -1,
            degreeShift: 50
        });

        const swirlL1 = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '0%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -100},
            radius: 30,
            swirlSize: 30,
            swirlFrequency: 1,
            duration: 1000,
            direction: -1,
            degreeShift: -90
        });
        const swirlL2 = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '0%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -85},
            radius: 30,
            swirlSize: 30,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: -10
        });
        const swirlL3 = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '0%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -70},
            radius: 30,
            swirlSize: 30,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: -30
        });


        /* RIGHT SIDE */
        const swirlR1B = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '100%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -100},
            radius: 30,
            swirlSize: 5,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: -90
        });
        const swirlR2B = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '100%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -85},
            radius: 25,
            swirlSize: 5,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: -70
        });
        const swirlR3B = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '100%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -70},
            radius: 20,
            swirlSize: 5,
            swirlFrequency: 1,
            duration: 1000,
            direction: -1,
            degreeShift: -50
        });

        const swirlL1B = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '100%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -100},
            radius: 30,
            swirlSize: 30,
            swirlFrequency: 1,
            duration: 1000,
            direction: -1,
            degreeShift: 90
        });
        const swirlL2B = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '100%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -85},
            radius: 30,
            swirlSize: 30,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: 10
        });
        const swirlL3B = new mojs.ShapeSwirl({
            parent: deleteItem,
            top: '100%',
            left: '100%',
            fill: 'rgba(255,255,255,1)',
            y: {0: -70},
            radius: 30,
            swirlSize: 30,
            swirlFrequency: 1,
            duration: 500,
            direction: -1,
            degreeShift: 30
        });

        const timeline = new mojs.Timeline;

        timeline.add(swirlR1, swirlR2, swirlR3, swirlL1, swirlL2, swirlL3, swirlR1B, swirlR2B, swirlR3B, swirlL1B, swirlL2B, swirlL3B);

        timeline.play();
    }
})