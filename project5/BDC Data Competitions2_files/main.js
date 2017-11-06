var comp = {};

//Hook listner on fileselect 
$(document).on('change', ':file', function() {
    var input = $(this),
        numFiles = input.get(0).files ? input.get(0).files.length : 1,
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [numFiles, label]);
});


//Start
$(document).ready(function() {

  //Custom upload file btn
  comp.fileUploadBtn();

  //Setup datatables
  if($('#thedatatable').length)
  {
    var t = $('#thedatatable').DataTable({});
  }
  
  //Setup dropdown menu
  $(".dropdown-toggle").dropdown();
  
  //Setup member remove modal
  comp.memberRemoveModalHandler();
 
  //Set up request deny modal
  comp.requestDenyModalHandler();

  //Setup join team modal cancel button 
  comp.joinTeamModalCancelBtnHandler();

});


comp.fileUploadBtn = function(){
  $("#id_name").attr("placeholder","Name this submission");
  $("#id_name").addClass("form-control");
  $(':file').on('fileselect', function(event, numFiles, label) {
        $("#upload-file-input-text").empty();
        $("#upload-file-input-text").addClass("btn-file-text");
        $("#upload-file-input-text").text(label);
  });
};


comp.memberRemoveModalHandler = function(){
  $('#memberRemoveModal').on('show.bs.modal', function (event) { 
    var url = $(event.relatedTarget).data('theremoveurl'); 
    $(this).find("#memberRemoveModalBtn").attr("href", url);
  });
};


comp.requestDenyModalHandler = function(){
  $('#requestDenyModal').on('show.bs.modal', function (event) { 
    var url = $(event.relatedTarget).data('theremoveurl'); 
    $(this).find("#requestDenyModalBtn").attr("href", url);
  });
};


comp.joinTeamModalCancelBtnHandler = function(){
  var savedV="";
  
  $('#joinTeamModal').on('show.bs.modal', function (event) { 
    savedV = $(this).find("#id_team_affiliation").val();
  });

  $(".joinTeamModalCancelBtn").click(function(){
    $("#id_team_affiliation").val(savedV);
  });

  $("#joinTeamModalBtn").click(function(){
    if($("#id_team_affiliation").val()!="")
    {
      var t = $("#id_team_affiliation option:selected").text();
      $("#profile-team-join-btn").text(t);
      $("#profile-team-option-or").hide();
      $("#profile-team-create-btn").hide();
    }
    else
    {
      $("#profile-team-join-btn").text("Join an exiting team");
      $("#profile-team-option-or").show();
      $("#profile-team-create-btn").show();
    }
  });

};


//Handler for creat team modal submit button
comp.cretteTeamSubmit = function() {
  var f=$("#create-team-iframe").contents().find("#create-team-form");
  var name = $(f).find("#id_name").val();
  var desc = $(f).find("#id_description").val();
  
  $("#team-btns-holder").hide();
  $("#temp-team-afflication-input").val(name);
  $("#temp-team-afflication-holder").show();

  setTimeout(function(){ 
    var el = $("#create-team-iframe").contents().find(".errorlist").length;
    if(el==0)
      $("#createTeamModal").modal('toggle');
  }, 2000);
};

//Set the id_team_affiliation field value from the create team form
comp.setTeamAfficationVal = function(val) {
  $('#id_team_affiliation').append($("<option></option>").attr("value",val).text(val)); 
  $('#id_team_affiliation').val(val);
};




