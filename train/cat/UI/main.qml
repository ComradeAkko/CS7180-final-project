import QtQuick
import QtQuick.Controls.Basic

ApplicationWindow {
    visible: true
    width: 600
    height: 500
    title: "HelloApp" 

    property string currTime: "00:00:00"
    property QtObject backend
    Rectangle {
        anchors.fill: parent        
        Text {
            anchors {
                bottom: parent.bottom
                bottomMargin: 12
                left: parent.left
                leftMargin: 12
            }
            text: currTime
            font.pixelSize: 48
            color: "black"
        }    
    }
    Connections {
        target: backend        
        function onUpdated(msg) {
            currTime = msg;
        }
    }
}